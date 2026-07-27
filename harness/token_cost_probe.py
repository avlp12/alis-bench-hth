#!/usr/bin/env python3
"""token_cost_probe.py — measure tokens/item PER TASK at the real settings.

The 770-item plan was priced by counting items; items differ by an order of
magnitude in generation cost, which is why the plan looked affordable and was
not. This prices each candidate task in tokens, from the model's own behaviour:
N seeded-random items per task, sent to the live endpoint with the exact system
instruction, budget and sampling the scored run would use.

Run under .venv-lmeval/bin/python (needs lm_eval task API for the prompts).

Doubles as the c=8 validation at REAL generation lengths: the aggregate tok/s
realized here is the number the 2048-token probe could not predict.
"""
import argparse, json, random, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

from lm_eval.tasks import TaskManager

FORMAT_INSTRUCTION = "End your reply with a final line of the form: The answer is <answer>."
TASKS = ["gpqa_diamond_cot_zeroshot", "aime25", "minerva_math500", "ifeval", "mmlu_pro"]
NO_SYSTEM = {"ifeval"}   # scores per-item instructions; a global one corrupts it


def leaves(o):
    """Yield only scoreable leaf tasks. tm.load() returns {'tasks':…,'groups':…};
    group entries are Group objects with no docs of their own — skip them."""
    if isinstance(o, dict):
        for v in o.values():
            yield from leaves(v)
    elif hasattr(o, "has_test_docs"):
        yield o


def collect(k, seed):
    """[(top_task, leaf_task, prompt_text)] — k seeded-random items per top task."""
    tm = TaskManager()
    rng = random.Random(seed)
    items = []
    for t in TASKS:
        pool = []
        for task in leaves(tm.load([t]) if hasattr(tm, "load") else tm.load_task_or_group([t])):
            docs = list(task.test_docs()) if task.has_test_docs() else list(task.validation_docs())
            name = task.config.task if hasattr(task.config, "task") else t
            for i in rng.sample(range(len(docs)), min(3, len(docs))):
                pool.append((t, name, str(task.doc_to_text(docs[i]))))
        items += rng.sample(pool, min(k, len(pool)))
    rng.shuffle(items)   # spread heavy tasks across concurrency waves
    return items


def ask(url, model, temp, top_p, max_tokens, timeout, item):
    top, leaf, prompt = item
    msgs = ([] if top in NO_SYSTEM else [{"role": "system", "content": FORMAT_INSTRUCTION}])
    msgs.append({"role": "user", "content": prompt})
    body = json.dumps({"model": model, "messages": msgs, "max_tokens": max_tokens,
                       "temperature": temp, "top_p": top_p}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            j = json.loads(r.read())
        u, msg = j.get("usage") or {}, (j.get("choices") or [{}])[0].get("message") or {}
        row = {"task": top, "leaf": leaf, "secs": round(time.time() - t0, 1),
               "completion_tokens": int(u.get("completion_tokens") or 0),
               "empty": not (msg.get("content") or "").strip()}
    except Exception as e:
        row = {"task": top, "leaf": leaf, "secs": round(time.time() - t0, 1),
               "completion_tokens": 0, "empty": True, "err": f"{type(e).__name__}: {e}"[:90]}
    print(f"  {row['task']:28s} {row['completion_tokens']:>6d} tok  {row['secs']/60:>5.1f} min"
          f"  {'EMPTY' if row['empty'] else 'ok'}", flush=True)
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True); p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--k", type=int, default=3); p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--temp", type=float, default=1.0); p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--timeout", type=int, default=10800)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="")
    a = p.parse_args()
    items = collect(a.k, a.seed)
    print(f"[cost] {a.name}: {len(items)} items, c={a.concurrency}, max_tokens={a.max_tokens}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        rows = list(ex.map(lambda it: ask(a.url, a.model, a.temp, a.top_p,
                                          a.max_tokens, a.timeout, it), items))
    wall = time.time() - t0
    per = {}
    for t in TASKS:
        rs = [r for r in rows if r["task"] == t]
        toks = sorted(r["completion_tokens"] for r in rs)
        per[t] = {"n": len(rs), "mean_tokens": round(sum(toks) / len(rs)) if rs else 0,
                  "median_tokens": toks[len(toks) // 2] if toks else 0,
                  "max_tokens_seen": toks[-1] if toks else 0,
                  "empty": sum(1 for r in rs if r["empty"]),
                  "mean_secs": round(sum(r["secs"] for r in rs) / len(rs), 1) if rs else 0}
    out = {"name": a.name, "budget": a.max_tokens, "concurrency": a.concurrency,
           "seed": a.seed, "wall_min": round(wall / 60, 1),
           "aggregate_tok_s": round(sum(r["completion_tokens"] for r in rows) / wall, 2),
           "per_task": per, "rows": rows}
    print(f"[cost] {a.name}: wall={out['wall_min']} min  realized aggregate={out['aggregate_tok_s']} tok/s", flush=True)
    for t, v in per.items():
        print(f"  {t:28s} mean={v['mean_tokens']:>6d}  median={v['median_tokens']:>6d}"
              f"  max={v['max_tokens_seen']:>6d}  empty={v['empty']}/{v['n']}", flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"[cost] wrote {a.out}", flush=True)
