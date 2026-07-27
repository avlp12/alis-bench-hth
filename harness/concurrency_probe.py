#!/usr/bin/env python3
"""concurrency_probe.py — does raising num_concurrent buy aggregate throughput?

Measured during R3: each individual request took ~42 min and 4 concurrent
requests completed together, giving ~10.5 min/item and a ~5.6-day projection per
leg. Before cutting the preregistered sample size, find out whether the servers
have headroom — MoE decode reads 8 of 384 experts per token, so a larger batch
can amortise those weight reads over more tokens and raise aggregate tok/s at
little cost to per-stream latency.

What is measured: AGGREGATE tokens/second at each concurrency level, i.e. the
only quantity that sets wall clock for a benchmark of N independent items.
Per-stream latency is reported too, because it is what rises as batch grows.

Prompts are DIFFERENT per request on purpose: mlx_lm.server caches prompts, and
identical prompts would measure the cache rather than the model.

  python3 concurrency_probe.py --name solar --url http://127.0.0.1:8082 \
      --model /path/to/build --levels 4,8,16 --max-tokens 2048
"""
import argparse, json, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

SYS = "End your reply with a final line of the form: The answer is <answer>."
# Reasoning-heavy, mmlu_pro-shaped, and distinct per index so nothing is cached.
STEMS = [
    "A cylindrical tank of radius {r} m is filled at {q} m^3/s. How fast does the level rise?",
    "A gas at {r} atm and {q}00 K expands adiabatically to half its volume. Find the final temperature.",
    "A loan of {r}0000 at {q}% compounded monthly is repaid over 10 years. Find the monthly payment.",
    "A population grows logistically with r={q}.1 and K={r}000, starting at 50. When does it reach K/2?",
    "Light of wavelength {r}00 nm strikes a metal of work function {q}.1 eV. Find the stopping potential.",
    "A beam of length {r} m carries a point load of {q} kN at midspan. Find the maximum bending moment.",
]

def one(url, model, temp, top_p, max_tokens, idx, timeout):
    stem = STEMS[idx % len(STEMS)].format(r=3 + idx % 7, q=1 + idx % 9)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": f"Question {idx}: {stem}\nThink step by step."}],
        "max_tokens": max_tokens, "temperature": temp, "top_p": top_p,
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            j = json.loads(r.read())
    except Exception as e:
        return {"ok": False, "err": f"{type(e).__name__}: {e}"[:90], "secs": time.time() - t0,
                "tokens": 0, "empty": True}
    dt = time.time() - t0
    u = j.get("usage") or {}
    msg = (j.get("choices") or [{}])[0].get("message") or {}
    return {"ok": True, "secs": dt, "tokens": int(u.get("completion_tokens") or 0),
            "empty": not (msg.get("content") or "").strip()}

def level(name, url, model, temp, top_p, max_tokens, c, timeout):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=c) as ex:
        rs = list(ex.map(lambda i: one(url, model, temp, top_p, max_tokens, i, timeout), range(c)))
    wall = time.time() - t0
    okr = [r for r in rs if r["ok"]]
    tok = sum(r["tokens"] for r in okr)
    agg = tok / wall if wall else 0.0
    per = (sum(r["secs"] for r in okr) / len(okr)) if okr else 0.0
    row = {"concurrency": c, "wall_s": round(wall, 1), "completed": len(okr), "failed": len(rs) - len(okr),
           "total_tokens": tok, "aggregate_tok_s": round(agg, 2),
           "per_stream_tok_s": round(tok / len(okr) / per, 2) if okr and per else 0.0,
           "mean_request_s": round(per, 1),
           "empty_content": sum(1 for r in okr if r["empty"]),
           "errors": sorted({r["err"] for r in rs if not r["ok"]})[:3]}
    print(f"  c={c:<3} agg={row['aggregate_tok_s']:>7.2f} tok/s  "
          f"per-stream={row['per_stream_tok_s']:>5.2f}  wall={wall/60:>5.1f} min  "
          f"ok={len(okr)}/{c}  empty={row['empty_content']}", flush=True)
    return row

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True); p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--levels", default="4,8,16")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temp", type=float, default=1.0); p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--timeout", type=int, default=5400)
    p.add_argument("--out", default="")
    a = p.parse_args()
    print(f"[conc] {a.name} {a.url} max_tokens={a.max_tokens} levels={a.levels}", flush=True)
    rows = [level(a.name, a.url, a.model, a.temp, a.top_p, a.max_tokens, int(c), a.timeout)
            for c in a.levels.split(",")]
    best = max(rows, key=lambda r: r["aggregate_tok_s"])
    base = rows[0]
    out = {"name": a.name, "url": a.url, "max_tokens": a.max_tokens, "levels": rows,
           "best_concurrency": best["concurrency"],
           "speedup_vs_first": round(best["aggregate_tok_s"] / base["aggregate_tok_s"], 2)
                               if base["aggregate_tok_s"] else None,
           # per-stream latency is what a human waiting on ONE answer feels; it is
           # expected to DROP as batch grows. That is an acceptable trade for a
           # benchmark (N independent items) and a bad one for interactive use.
           "note": "aggregate tok/s sets benchmark wall clock; per-stream tok/s is latency"}
    print(f"[conc] {a.name}: best c={out['best_concurrency']} "
          f"({out['speedup_vs_first']}x vs c={base['concurrency']})", flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"[conc] wrote {a.out}", flush=True)
