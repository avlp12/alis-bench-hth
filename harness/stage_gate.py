#!/usr/bin/env python3
"""Stage gate — verify every layer of the pipeline in ONE pass, cheaply.

Why this exists: defects in a layered pipeline are discovered serially, because
each broken layer hides the next. Dependencies must import before a task can
load; a task must load before a backend is called; a backend must answer before
extraction can be judged; extraction must work before scoring means anything. So
"run → hit wall → fix → rerun" finds exactly one defect per expensive run, and
the dangerous ones live deepest (a wrong number is worse than a crash).

This walks all layers in order, each with the smallest possible probe, and
reports EVERY failure it can reach rather than stopping at the first. Run it
before any scored run; it costs minutes, not hours.

  python3 stage_gate.py            # all stages
  python3 stage_gate.py --skip-gen # config/deps only (no model calls)
"""
import argparse, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
R = [];  FAIL = 0

def stage(n, title):
    print(f"\n\033[1m== stage {n}: {title} ==\033[0m")

def ok(m):   print(f"  \033[32m✓\033[0m {m}")
def bad(m):
    global FAIL; FAIL += 1; print(f"  \033[31m✗\033[0m {m}")
def warn(m): print(f"  \033[33m!\033[0m {m}")

def env(k, d=""): return os.environ.get(k, d)

# ---------------------------------------------------------------- stage 1
def s1_imports():
    stage(1, "interpreters and imports (a missing dep hides every later layer)")
    lm = HERE.parent / ".venv-lmeval/bin/python"
    if lm.exists():
        r = subprocess.run([str(lm), "-c",
            "import lm_eval,langdetect,immutabledict,sympy,math_verify,tenacity;"
            "import lm_eval.tasks;"
            "import huggingface_hub as h;print(h.__version__)"],
            capture_output=True, text=True)
        ok(f"lm-eval venv complete (hub {r.stdout.strip()})") if r.returncode == 0 \
            else bad(f"lm-eval venv missing deps: {r.stderr.strip().splitlines()[-1][:140]}")
    else:
        bad(f"lm-eval venv absent at {lm}")
    r = subprocess.run([sys.executable, "-c",
        "import llm_eval, litellm, spacy; spacy.load('ko_core_news_sm'); print('ok')"],
        capture_output=True, text=True)
    ok("HRET stack imports (incl. spaCy ko)") if r.returncode == 0 \
        else bad(f"HRET stack: {r.stderr.strip().splitlines()[-1][:140]}")

# ---------------------------------------------------------------- stage 2
def s2_tasks():
    stage(2, "every scored task actually loads (else it dies mid-run)")
    lm = HERE.parent / ".venv-lmeval/bin/python"
    tasks = os.environ.get("TASKS", "mmlu_pro,bbh_cot_zeroshot,gsm8k_cot_zeroshot,minerva_math") + ",ifeval"
    code = ("from lm_eval.tasks import TaskManager;tm=TaskManager()\n"
            "import sys\n"
            f"ts='{tasks}'.split(',')\n"
            "bad=[]\n"
            "for t in ts:\n"
            "    try:\n"
            "        r = tm.load([t]) if hasattr(tm,'load') else tm.load_task_or_group([t])\n"
            "        assert (r.get('tasks') or r.get('groups')) if isinstance(r,dict) else r\n"
            "    except Exception as e: bad.append(f'{t}: {type(e).__name__}')\n"
            "print('BAD:'+';'.join(bad) if bad else 'ALLOK')\n")
    r = subprocess.run([str(lm), "-c", code], capture_output=True, text=True)
    out = [l for l in r.stdout.splitlines() if l.startswith(("BAD:", "ALLOK"))]
    if out and out[-1] == "ALLOK": ok(f"all lm-eval tasks load: {tasks}")
    else: bad(f"lm-eval task load: {out[-1] if out else r.stderr[-160:]}")

    code2 = ("from llm_eval.datasets import DATASET_REGISTRY as D\n"
             "want=['kmmlu','haerae_bench','hrm8k','click','kormedmcqa']\n"
             "miss=[w for w in want if w not in D]\n"
             "print('MISS:'+','.join(miss) if miss else 'ALLOK')\n")
    r = subprocess.run([sys.executable, "-c", code2], capture_output=True, text=True)
    out = [l for l in r.stdout.splitlines() if l.startswith(("MISS:", "ALLOK"))]
    if out and out[-1] == "ALLOK": ok("all HRET dataset keys exist in the registry")
    else: bad(f"HRET dataset keys: {out[-1] if out else r.stderr[-160:]}")

# ---------------------------------------------------------------- stage 3
def _chat(url, model, prompt, mx=256, temp=1.0, top_p=0.95, timeout=1800):
    b = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": mx, "temperature": temp, "top_p": top_p}).encode()
    rq = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=b,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=timeout) as f:
        return json.load(f)

def s3_endpoints():
    stage(3, "each endpoint answers, and the answer is an ANSWER (not an error string)")
    for side in ("motif", "solar"):
        url, model = env(f"{side.upper()}_URL"), env(f"{side.upper()}_MODEL")
        if not url or not model: bad(f"{side}: URL/MODEL unset"); continue
        try:
            t0 = time.time()
            j = _chat(url, model, "1+1은 얼마인가? 숫자만 답하라.")
            msg = (j.get("choices") or [{}])[0].get("message") or {}
            c = (msg.get("content") or "").strip()
            think = (msg.get("reasoning_content") or msg.get("reasoning") or "")
            if j.get("error"): bad(f"{side}: server error {str(j['error'])[:100]}")
            elif c.startswith("Error:"): bad(f"{side}: backend returned an ERROR STRING as the answer")
            elif not c and think: bad(f"{side}: empty content with {len(think)} chars reasoning — raise MAX_TOKENS")
            elif "2" not in c: bad(f"{side}: wrong answer {c[:60]!r}")
            else: ok(f"{side}: {c[:40]!r} ({time.time()-t0:.0f}s, reasoning {len(think)}c)")
            for m in ("</think>", "<|think:end|>", "<think>", "<|think:start|>"):
                if m in c: bad(f"{side}: think marker {m!r} leaked into content — extraction would score the reasoning")
        except Exception as e:
            bad(f"{side}: {type(e).__name__}: {str(e)[:120]}")

# ---------------------------------------------------------------- stage 4
def s4_hret_backend():
    stage(4, "HRET backend wiring (the layer that silently returned error strings)")
    try:
        from llm_eval.models import MODEL_REGISTRY
    except Exception as e:
        bad(f"HRET import: {e}"); return
    B = MODEL_REGISTRY.get("litellm")
    for side in ("motif", "solar"):
        url, model = env(f"{side.upper()}_URL"), env(f"{side.upper()}_MODEL")
        if not url: continue
        try:
            be = B(provider="openai", model_name=f"openai/{model}",
                   api_base=url.rstrip("/") + "/v1", api_key="sk-noauth",
                   temperature=1.0, max_new_tokens=128)
            out = be.generate_batch([{"input": "1+1은? 숫자만.", "reference": "2"}])
            pred = str(out[0].get("prediction") or "").strip()
            if pred.startswith("Error:"): bad(f"{side}: HRET backend -> {pred[:110]!r}")
            elif not pred: bad(f"{side}: HRET backend returned empty prediction")
            else: ok(f"{side}: HRET backend -> {pred[:50]!r}")
        except TypeError as e:
            bad(f"{side}: HRET backend signature: {str(e)[:120]}")
        except Exception as e:
            bad(f"{side}: HRET backend: {type(e).__name__}: {str(e)[:120]}")

# ---------------------------------------------------------------- stage 5
def s5_aggregate():
    stage(5, "aggregation + verdict path on synthetic-but-realistic data")
    import tempfile, random
    d = Path(tempfile.mkdtemp())
    random.seed(0)
    for name, adv in (("motif", 0.15), ("solar", 0.0)):
        p = d / "lm_eval" / name / "run"; p.mkdir(parents=True)
        json.dump({"results": {}, "groups": {"mmlu_pro": {
            "exact_match,custom-extract": 0.5 + adv, "exact_match_stderr,none": 0.02}}},
            open(p / "results_x.json", "w"))
        with open(p / "samples_mmlu_pro_2026-01-01.jsonl", "w") as f:
            for i in range(120):
                f.write(json.dumps({"doc_id": i, "filter": "custom-extract",
                                    "exact_match": float(random.random() < 0.5 + adv),
                                    "resps": [["ok"]]}) + "\n")
        h = d / "hret" / name; h.mkdir(parents=True)
        json.dump({"kmmlu": {"raw": {"metrics": {"accuracy": 0.6 + adv}}, "_primary": "raw"}},
                  open(h / "_summary.json", "w"))
        # exercise the KOREAN PAIRED path — the gate previously passed while this
        # code persisted nothing at all on the real HRET
        json.dump([{"key": f"it{i}", "score": float(random.random() < 0.6 + adv)}
                   for i in range(60)], open(h / "kmmlu_items.json", "w"))
    # exercise the JUDGE length-control path
    j = d / "judge"; j.mkdir(parents=True)
    rows = [{"final": "motif" if i % 3 else "solar",
             "answer_motif": "m" * (200 + 8 * i), "answer_solar": "s" * 200} for i in range(40)]
    json.dump({"summary": {"n": 40, "motif_wins": 27, "solar_wins": 13, "semantic_ties": 0,
                           "judges": ["a", "b"], "verdict": "motif"}, "rows": rows},
              open(j / "pairwise.json", "w"))
    envv = dict(os.environ, RESULTS_DIR=str(d), BPW_TIER="gate", ALLOW_PARTIAL="1",
                JUDGE_ENDPOINTS="", MOTIF_REF="", SOLAR_REF="")
    subprocess.run([sys.executable, str(HERE / "manifest.py"), "init"], env=envv,
                   capture_output=True, text=True)
    r = subprocess.run([sys.executable, str(HERE / "aggregate.py")], env=envv,
                       capture_output=True, text=True)
    txt = r.stdout
    if r.returncode != 0: bad(f"aggregate crashed: {r.stderr.strip().splitlines()[-1][:140]}")
    else:
        checks = {"paired verdict emitted": "mmlu_pro" in txt and ("motif" in txt or "INCONCLUSIVE" in txt),
                  "HRET metrics parsed (not '—')": "hret:kmmlu" in txt and "0.6" in txt.replace("0.60", "0.6"),
                  "plumbing-failure section present": "Plumbing failures" in txt,
                  "KOREAN paired row produced": "hret:kmmlu" in txt and "n=" in txt,
                  "disclosed-cost section present": "Disclosed cost" in txt}
        for k, v in checks.items(): ok(k) if v else bad(f"aggregate: {k} FAILED")
    import shutil; shutil.rmtree(d, ignore_errors=True)

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-gen", action="store_true")
    a = ap.parse_args()
    print("\033[1mSTAGE GATE\033[0m — every layer, one pass, cheapest probe per layer")
    s1_imports(); s2_tasks()
    if not a.skip_gen:
        s3_endpoints(); s4_hret_backend()
    else:
        warn("stages 3-4 skipped (--skip-gen): no model calls made")
    s5_aggregate()
    print()
    if FAIL: print(f"\033[31mSTAGE GATE FAIL: {FAIL} blocking issue(s)\033[0m — fix all, then re-run this gate."); sys.exit(1)
    print("\033[32mSTAGE GATE PASS\033[0m — every layer verified; the scored run may start.")
