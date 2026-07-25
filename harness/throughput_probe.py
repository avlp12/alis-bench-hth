#!/usr/bin/env python3
"""Efficiency probe — the ONLY part that must run on ONE box (serve both models
back-to-back on the same machine; cross-box latency numbers are not comparable).

Reports, per language slice (KO / EN / code):
  * TTFT           — time to first token (streaming)
  * decode_tok_s   — output tokens / (total - ttft)
  * chars_s        — output CHARS / (total - ttft)   <-- TOKENIZER-NEUTRAL
  * prefill_tok_s  — prompt tokens / ttft (approx)

Why chars_s: Solar's Korean-optimized tokenizer emits ~50-80% of the tokens for
the same Korean text, so tokens/s flatters Motif and understates Solar. chars/s
(useful output per second) is the fair cross-tokenizer efficiency metric.

Usage (same box):
  ./serve.sh motif 2>/dev/null &   # or point --url at whichever is up
  python3 throughput_probe.py --name motif --url http://localhost:8081
  # then stop it, serve solar, and:
  python3 throughput_probe.py --name solar --url http://localhost:8082
"""
import argparse, json, os, time, urllib.request
from pathlib import Path

PROMPTS = {
    "KO":  "인공지능 반도체 산업의 최근 동향과 한국 기업의 경쟁력을 자세히 분석해줘.",
    "EN":  "Explain the trade-offs between mixture-of-experts and dense transformers in detail.",
    "code":"Write a Python class implementing an LRU cache with get/put in O(1), with tests.",
}

def stream(url, model, prompt, max_tokens, temp=0.7):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": temp, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; text = []; usage = {}
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
            except Exception:
                continue
            if j.get("usage"):
                usage = j["usage"]
            ch = (j.get("choices") or [{}])[0]
            delta = (ch.get("delta") or {}).get("content") or ""
            if delta and ttft is None:
                ttft = time.perf_counter() - t0
            text.append(delta)
    total = time.perf_counter() - t0
    return ttft or total, total, "".join(text), usage

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)      # motif | solar (label only)
    ap.add_argument("--url", required=True)       # endpoint on THIS box
    ap.add_argument("--model", default=None)      # served model id
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    model = a.model or os.environ.get(f"{a.name.upper()}_MODEL", a.name)
    # Warm-up (discarded): first call pays weight paging, kernel JIT and a cold
    # clock ramp. Without it the model measured FIRST is systematically penalised.
    print("[thru] warm-up (discarded) ...")
    try:
        stream(a.url, model, PROMPTS["EN"], 64)
    except Exception as e:
        print(f"[thru] warm-up failed: {e}")
    out = {"_meta": {"warmup": True, "reps": a.reps, "max_tokens": a.max_tokens,
                     "note": "run both models on the SAME box, same thermal state; "
                             "report chars_s AND decode_tok_s — neither is unbiased"}}
    for lang, prompt in PROMPTS.items():
        runs = []
        for i in range(a.reps):
            ttft, total, text, usage = stream(a.url, model, prompt, a.max_tokens)
            ot = usage.get("completion_tokens") or 0
            pt = usage.get("prompt_tokens") or 0
            dec = max(total - ttft, 1e-6)
            runs.append({
                "ttft_s": round(ttft, 3),
                "decode_tok_s": round(ot / dec, 1) if ot else None,
                "chars_s": round(len(text) / dec, 1),
                "prefill_tok_s": round(pt / ttft, 1) if (pt and ttft) else None,
                "out_tokens": ot, "out_chars": len(text),
            })
        # median run + full spread retained (a single number hides thermal drift)
        runs_sorted = sorted(runs, key=lambda r: r["chars_s"])
        med = dict(runs_sorted[len(runs_sorted) // 2])
        med["all_runs"] = runs
        med["chars_s_min_max"] = [runs_sorted[0]["chars_s"], runs_sorted[-1]["chars_s"]]
        out[lang] = med
        print(f"[thru] {a.name} {lang}: chars_s={med['chars_s']} "
              f"(range {med['chars_s_min_max']}) tok_s={med['decode_tok_s']} ttft={med['ttft_s']}")
    outdir = Path(os.environ.get("RESULTS_DIR", ".")) / "throughput"; outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{a.name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[thru] {a.name} -> {outdir}/{a.name}.json  (report chars_s AND tok/s; neither is clean)")
    # self-stamp into the current run so the integrity gate sees a fresh throughput cell
    man = Path(os.environ.get("RESULTS_DIR", ".")) / "run_manifest.json"
    if man.exists():
        import subprocess, sys
        mp = Path(__file__).parent / "manifest.py"
        subprocess.run([sys.executable, str(mp), "record", f"thru.{a.name}", "0"])
        subprocess.run([sys.executable, str(mp), "stamp", f"throughput/{a.name}.json"])
        print(f"[thru] stamped thru.{a.name} into the run")

if __name__ == "__main__":
    main()
