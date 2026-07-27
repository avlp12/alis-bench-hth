#!/usr/bin/env python3
"""Korean open-generation quality via pairwise LLM-judge — hardened.

Fixes from adversarial review:
  * >=2 INDEPENDENT judge families (JUDGE_ENDPOINTS); a win requires judge
    agreement, and per-judge position-bias is reported (a biased judge is flagged).
  * Ties, parse-failures, and order-effects are counted SEPARATELY — never
    laundered into "vs decided". The headline win-rate is over ALL n.
  * Raw retained: prompts, both answers, every judge's raw output.
  * Publishable gate: the built-in 5-item smoke set (or SUITE_MODE=smoke) marks
    the result SMOKE / not-publishable; a real run needs --prompts <full corpus>.
  * turn2 (multi-turn) is evaluated when present.
  * Overall verdict = paired sign-test bootstrap over items with an equivalence
    margin -> motif / solar / INCONCLUSIVE.

JUDGE_ENDPOINTS (JSON list), each an OpenAI-compatible /chat/completions endpoint:
  [{"name":"gpt","model":"gpt-5.1","base_url":"https://api.openai.com/v1","key_env":"OPENAI_API_KEY"},
   {"name":"ko-specialist","model":"...","base_url":"https://.../v1","key_env":"KO_JUDGE_KEY"}]
Neither judge may be a contender. Falls back to the single legacy JUDGE_* (flagged not-publishable).
"""
import argparse, json, os, sys, urllib.request
from pathlib import Path
import numpy as np

BUILTIN = [
    {"id": "b1", "category": "추론", "turn1": "빨간 공 3개, 파란 공 5개가 있다. 파란 공 2개를 빨간 공으로 칠하면 각 색의 공은 몇 개인가? 풀이와 함께 답하라."},
    {"id": "b2", "category": "글쓰기", "turn1": "고향의 겨울 아침을 주제로 다섯 문장짜리 한국어 수필을 써라."},
    {"id": "b3", "category": "코딩", "turn1": "파이썬으로 문자열이 회문인지 판별하는 함수를 작성하고 한국어 주석을 달아라."},
    {"id": "b4", "category": "이해", "turn1": "다음 논증의 오류를 지적하라: '모든 새는 난다. 펭귄은 새다. 그러므로 펭귄은 난다.'"},
    {"id": "b5", "category": "문법", "turn1": "'되'와 '돼'의 차이를 설명하고 올바른 예문을 하나씩 들어라."},
]
JUDGE_SYS = ("당신은 두 한국어 AI 답변을 비교하는 엄격하고 공정한 심사위원이다. 정확성·유용성·지시이행·한국어 "
             "자연스러움으로 평가하라. 길이나 제시 순서에 편향되지 말라. 간단한 근거 후, 마지막 줄에 반드시 "
             "[[A]], [[B]], [[TIE]] 중 하나만 출력하라.")

def http(url, body, headers, timeout=1200):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def gen(url, model, messages, temp, top_p, seed, mx):
    """-> (text, tokens_spent). Token spend is the DISCLOSED COST of running a
    model at its peak; without it 'best-vs-best with disclosed cost' is a claim
    we cannot back, and more-compute-wins is indistinguishable from better."""
    r = http(url.rstrip("/") + "/v1/chat/completions",
             {"model": model, "messages": messages, "temperature": temp,
              "top_p": top_p, "seed": seed, "max_tokens": mx, "stream": False}, {})
    u = r.get("usage") or {}
    ch = r["choices"][0]["message"]
    # some servers expose reasoning/thinking tokens separately
    reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return ch["content"].strip(), {"completion_tokens": u.get("completion_tokens"),
                                   "reasoning_tokens": reasoning,
                                   "chars": len(ch.get("content") or "")}

def judge_once(ep, question, ans_a, ans_b):
    content = f"[질문]\n{question}\n\n[답변 A]\n{ans_a}\n\n[답변 B]\n{ans_b}\n\n위 기준으로 더 나은 답변을 고르라."
    try:
        r = http(ep["base_url"].rstrip("/") + "/chat/completions",
                 {"model": ep["model"], "temperature": 0.0,
                  "messages": [{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": content}]},
                 {"Authorization": f"Bearer {os.environ.get(ep.get('key_env',''), ep.get('key',''))}"})
        txt = r["choices"][0]["message"]["content"]
    except Exception as e:
        return "FAIL", f"[error] {e}"
    v = txt.rsplit("[[", 1)[-1].split("]]")[0].strip().upper() if "[[" in txt else "FAIL"
    return (v if v in ("A", "B", "TIE") else "FAIL"), txt

def _ctx(prior, q, a_is_motif):
    """Render the conversation so far CONSISTENTLY with the current A/B assignment.
    (If the prior turns were always labelled A=motif while the current answers are
    swapped, the judge sees contradictory labels — that reintroduces the very
    position bias the two-order design removes.)"""
    if not prior:
        return q
    blocks = []
    for pq, pm, ps in prior:
        a_ans, b_ans = (pm, ps) if a_is_motif else (ps, pm)
        blocks.append(f"[이전 턴 질문]\n{pq}\n[A 이전 답변]\n{a_ans}\n[B 이전 답변]\n{b_ans}")
    return "\n\n".join(blocks) + f"\n\n[현재 턴 질문 — 이 턴의 답변만 평가하라]\n{q}"

def judge_item(ep, q, a_motif, a_solar, prior=()):
    """Run BOTH orders on one judge -> per-judge outcome + raw."""
    v1, t1 = judge_once(ep, _ctx(prior, q, True), a_motif, a_solar)   # A=motif
    v2, t2 = judge_once(ep, _ctx(prior, q, False), a_solar, a_motif)  # A=solar (swapped)
    if "FAIL" in (v1, v2):                          return "parse_fail", (t1, t2)
    r1 = {"A": "motif", "B": "solar", "TIE": "tie"}[v1]
    r2 = {"A": "solar", "B": "motif", "TIE": "tie"}[v2]
    if r1 == r2 == "tie":                           return "tie", (t1, t2)
    if r1 == r2:                                     return r1, (t1, t2)     # order-consistent win
    return "order_effect", (t1, t2)

def combine(outcomes):
    """Across judges: unanimous win -> win; else classify."""
    wins = set(o for o in outcomes if o in ("motif", "solar"))
    if outcomes.count("parse_fail"):                 return "parse_fail"
    if len(wins) == 1 and all(o == list(wins)[0] for o in outcomes): return list(wins)[0]
    if wins:                                          return "judge_disagree"
    if outcomes.count("order_effect"):               return "order_effect"
    return "tie"

def length_controlled(rows, seed=1234):
    """AlpacaEval-2.0-style length control.

    Telling a judge "do not be biased by length" does NOT work — that is precisely
    why length-controlled win rates exist. The published remedy is statistical:
    regress the preference on the length difference and report the win rate the
    model would get at ZERO length difference.

    Returns (raw_winrate, length_controlled_winrate, mean_len_winner,
             mean_len_loser, beta_length). Reported ALONGSIDE the raw number,
     never instead of it — length may genuinely correlate with quality.
    """
    import numpy as np
    xs, ys, wl, ll = [], [], [], []
    for r in rows:
        f = r.get("final")
        if f not in ("motif", "solar"):
            continue
        lm, ls = len(r.get("answer_motif") or ""), len(r.get("answer_solar") or "")
        denom = (lm + ls) or 1
        xs.append((lm - ls) / denom)          # normalized length difference
        ys.append(1.0 if f == "motif" else 0.0)
        (wl if f == "motif" else ll).append(lm)
        (ll if f == "motif" else wl).append(ls)
    n = len(ys)
    if n < 8:
        return (None,) * 5
    X = np.column_stack([np.ones(n), np.asarray(xs)])
    y = np.asarray(ys)
    w = np.zeros(2)
    for _ in range(200):                       # IRLS-ish gradient steps
        p_ = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p_) / n
        H = (X * (p_ * (1 - p_))[:, None]).T @ X / n + 1e-6 * np.eye(2)
        w = w + np.linalg.solve(H, g)
    lc = float(1 / (1 + np.exp(-w[0])))        # prediction at length_diff = 0
    return (float(y.mean()), lc,
            float(np.mean(wl)) if wl else None, float(np.mean(ll)) if ll else None,
            float(w[1]))

def bootstrap_sign(scores, margin, n=2000, seed=1234):
    a = np.asarray(scores, float)
    if a.size == 0: return (None, None, None, "no-data")
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n, a.size))].mean(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    # Same preregistered power floor as aggregate.verdict(): a handful of prompts
    # can put the CI clear of the margin by luck, and aggregate.py quotes this
    # verdict verbatim into the report. Gate the DIRECTION, not the estimate.
    min_n = int(os.environ.get("MIN_PAIRED_N", "30"))
    if a.size < min_n:
        return float(a.mean()), lo, hi, f"underpowered(n={a.size})"
    vd = "motif" if lo > margin else "solar" if hi < -margin else "INCONCLUSIVE"
    return float(a.mean()), lo, hi, vd

def load_judges():
    raw = os.environ.get("JUDGE_ENDPOINTS", "").strip()
    if raw:
        eps = json.loads(raw)
    elif os.environ.get("JUDGE_BASE_URL"):
        eps = [{"name": "legacy", "model": os.environ["JUDGE_MODEL"],
                "base_url": os.environ["JUDGE_BASE_URL"], "key": os.environ.get("JUDGE_API_KEY", "")}]
    else:
        sys.exit("[judge] set JUDGE_ENDPOINTS (>=2 neutral judges) — see header")
    return eps

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prompts"); a = ap.parse_args()
    C = os.environ
    smoke = (not a.prompts) or C.get("SUITE_MODE") == "smoke"
    prompts = [json.loads(l) for l in open(a.prompts)] if a.prompts else BUILTIN
    eps = load_judges()
    if len(eps) < 2:
        print("[judge] WARNING: <2 judges -> result is NOT publishable (family-bias uncontrolled)", file=sys.stderr)
    m_url, s_url = C["MOTIF_URL"], C["SOLAR_URL"]; m_mod, s_mod = C["MOTIF_MODEL"], C["SOLAR_MODEL"]
    m_t, m_p = float(C.get("MOTIF_TEMP", 1.0)), float(C.get("MOTIF_TOP_P", .95))
    s_t, s_p = float(C.get("SOLAR_TEMP", .7)), float(C.get("SOLAR_TOP_P", .95))
    seed, mx = int(C.get("SEED", 1234)), int(C.get("MAX_TOKENS", 8192))
    marg = float(C.get("JUDGE_MARGIN", "0.0"))
    outdir = Path(C["RESULTS_DIR"]) / "judge"; outdir.mkdir(parents=True, exist_ok=True)

    rows, scores = [], []
    cost = {"motif": [], "solar": []}
    tally = dict(motif=0, solar=0, tie=0, order_effect=0, parse_fail=0, judge_disagree=0)
    per_judge_bias = {e["name"]: 0 for e in eps}
    for p in prompts:
        turns = [("t1", p["turn1"])] + ([("t2", p["turn2"])] if p.get("turn2") else [])
        hist_m, hist_s = [], []
        for ti, (tkey, q) in enumerate(turns):
            hist_m.append({"role": "user", "content": q}); hist_s.append({"role": "user", "content": q})
            am, um = gen(m_url, m_mod, hist_m, m_t, m_p, seed, mx); hist_m.append({"role": "assistant", "content": am})
            as_, us = gen(s_url, s_mod, hist_s, s_t, s_p, seed, mx); hist_s.append({"role": "assistant", "content": as_})
            cost["motif"].append(um); cost["solar"].append(us)
            # Multi-turn: the judge must see the conversation so far, otherwise a
            # turn-2 answer is judged out of context (a follow-up like "then make
            # it shorter" is meaningless alone). _ctx() renders it per A/B order.
            prior = tuple((turns[j][1], hist_m[2 * j + 1]["content"], hist_s[2 * j + 1]["content"])
                          for j in range(ti))
            outs, raws = [], {}
            for e in eps:
                o, (t1, t2) = judge_item(e, q, am, as_, prior)
                outs.append(o); raws[e["name"]] = {"order_motifA": t1, "order_solarA": t2}
                if o == "order_effect": per_judge_bias[e["name"]] += 1
            final = combine(outs)
            tally[final] = tally.get(final, 0) + 1
            scores.append(1 if final == "motif" else -1 if final == "solar" else 0)
            rows.append({"id": f"{p.get('id')}#{tkey}", "category": p.get("category"), "q": q,
                         "answer_motif": am, "answer_solar": as_, "per_judge": outs, "final": final, "raw": raws})
            print(f"[judge] {p.get('id')}#{tkey}: judges={outs} -> {final}")

    n = len(scores)
    mean, lo, hi, vd = bootstrap_sign(scores, marg, int(C.get("BOOTSTRAP_N", 2000)), seed)
    if smoke: vd = "SMOKE (not publishable)"
    def _cost(rs):
        ct = [r["completion_tokens"] for r in rs if r.get("completion_tokens")]
        rt = [r["reasoning_tokens"] for r in rs if r.get("reasoning_tokens")]
        ch = [r["chars"] for r in rs if r.get("chars")]
        f = lambda x: round(sum(x) / len(x), 1) if x else None
        return {"mean_completion_tokens": f(ct), "mean_reasoning_tokens": f(rt),
                "mean_chars": f(ch), "total_completion_tokens": sum(ct) if ct else None}
    summary = {"n": n, "motif_wins": tally["motif"], "solar_wins": tally["solar"],
               "disclosed_cost": {k: _cost(v) for k, v in cost.items()},
               "semantic_ties": tally["tie"], "order_effect": tally["order_effect"],
               "parse_fail": tally["parse_fail"], "judge_disagree": tally["judge_disagree"],
               "motif_winrate_overall": round(tally["motif"] / n, 3) if n else None,
               "solar_winrate_overall": round(tally["solar"] / n, 3) if n else None,
               "score_mean": mean, "score_ci95": [lo, hi], "verdict": vd,
               "judges": [e["name"] for e in eps],
               "judge_position_bias": {k: round(v / (n or 1), 3) for k, v in per_judge_bias.items()},
               "judge_agreement": round(1 - (tally["judge_disagree"] / n), 3) if n else None,
               "publishable": (not smoke) and len(eps) >= 2}
    raw, lc, mw, ml, beta = length_controlled(rows, seed)
    summary["length_control"] = {
        "raw_winrate_vs_decided": None if raw is None else round(raw, 3),
        "length_controlled_winrate": None if lc is None else round(lc, 3),
        "mean_chars_winner": mw, "mean_chars_loser": ml,
        "beta_length": None if beta is None else round(beta, 3),
        "note": ("AlpacaEval-2.0-style: win rate predicted at zero length difference. "
                 "Report ALONGSIDE the raw rate — length may correlate with real quality "
                 "(LMArena's own caveat), so a controlled number is a second view, not a "
                 "correction. beta_length > 0 means longer Motif answers won more often."),
    }
    (outdir / "pairwise.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    print("[judge] summary:", json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
