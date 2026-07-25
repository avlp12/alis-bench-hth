#!/usr/bin/env python3
"""Blinded judging pack — run open-generation head-to-head WITHOUT a judge API.

Why this exists: most of the suite (MCQA, rule-verified IFEval, answer-extracted
math, executed code) needs no judge at all. Only open-generation quality does.
Rather than block the whole run on judge API keys — or hand the verdict to a
judge with a conflict of interest — this generates a properly blinded pack that a
*human* (ideally a Korean native speaker) or any second judge can score offline.

Blinding actually implemented (not just claimed):
  * model identity replaced by neutral labels 가 / 나
  * per-item side assignment randomized from a recorded seed
  * the mapping is written to a SEPARATE key file, so the reviewer file alone
    cannot reveal which side is which
  * items shuffled so one model is not consistently first

  python3 blind_pack.py make  [--prompts logickor.jsonl] [--limit N]
      -> $RESULTS_DIR/judge/blind_pack.md    (give this to the reviewer)
      -> $RESULTS_DIR/judge/blind_key.json   (do NOT open until scored)

  # reviewer fills the verdict lines in blind_pack.md: 가 / 나 / 무승부
  python3 blind_pack.py score --reviewer "name" [--pack ...] [--key ...]
      -> $RESULTS_DIR/judge/pairwise.json    (same shape aggregate.py consumes)
"""
import argparse, json, os, random, re, sys, urllib.request
from pathlib import Path

BUILTIN = [
    {"id": "b1", "category": "추론", "turn1": "빨간 공 3개, 파란 공 5개가 있다. 파란 공 2개를 빨간 공으로 칠하면 각 색의 공은 몇 개인가? 풀이와 함께 답하라."},
    {"id": "b2", "category": "글쓰기", "turn1": "고향의 겨울 아침을 주제로 다섯 문장짜리 한국어 수필을 써라."},
    {"id": "b3", "category": "코딩", "turn1": "파이썬으로 문자열이 회문인지 판별하는 함수를 작성하고 한국어 주석을 달아라."},
    {"id": "b4", "category": "이해", "turn1": "다음 논증의 오류를 지적하라: '모든 새는 난다. 펭귄은 새다. 그러므로 펭귄은 난다.'"},
    {"id": "b5", "category": "문법", "turn1": "'되'와 '돼'의 차이를 설명하고 올바른 예문을 하나씩 들어라."},
]

def gen(url, model, messages, temp, top_p, seed, mx):
    body = json.dumps({"model": model, "messages": messages, "temperature": temp,
                       "top_p": top_p, "seed": seed, "max_tokens": mx, "stream": False}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        resp = json.load(r)
    if resp.get("error"):
        raise SystemExit(f"[blind] server error: {str(resp['error'])[:200]}\n"
                         "  -> `model` must exactly equal the served path.")
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    u = resp.get("usage") or {}
    text = (msg.get("content") or "").strip()
    if not text:
        think = (msg.get("reasoning_content") or msg.get("reasoning") or "")
        raise SystemExit(f"[blind] empty content ({len(think)} chars of reasoning) — raise MAX_TOKENS")
    return text, {"completion_tokens": u.get("completion_tokens"),
                  "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                  "chars": len(text)}

def make(a):
    C = os.environ
    prompts = [json.loads(l) for l in open(a.prompts)] if a.prompts else BUILTIN
    if a.limit: prompts = prompts[:a.limit]
    seed = int(C.get("SEED", 1234))
    rng = random.Random(seed)
    out = Path(C["RESULTS_DIR"]) / "judge"; out.mkdir(parents=True, exist_ok=True)
    mx = int(C.get("MAX_TOKENS", 8192))

    key, md, cost = [], [], {"motif": [], "solar": []}
    md += ["# 블라인드 심사 팩", "",
           "두 모델의 답변을 **가 / 나**로 익명화했습니다. 어느 쪽이 어느 모델인지는 별도 키 파일에",
           "있으며, 채점을 마치기 전에는 열지 마십시오. 항목마다 **정확성 · 유용성 · 지시 이행 ·",
           "한국어 자연스러움**을 기준으로 더 나은 쪽을 고르고, 마지막 줄의 `판정:` 뒤에",
           "`가`, `나`, `무승부` 중 하나를 적어 주세요. 길이나 제시 순서에 끌리지 마십시오.", "",
           f"- 항목 수: {len(prompts)} · seed: {seed}", "", "---", ""]

    items = list(enumerate(prompts))
    rng.shuffle(items)
    for n, (_, p) in enumerate(items, 1):
        q = p["turn1"]
        m_txt, m_u = gen(C["MOTIF_URL"], C["MOTIF_MODEL"], [{"role": "user", "content": q}],
                         float(C.get("MOTIF_TEMP", 1.0)), float(C.get("MOTIF_TOP_P", .95)), seed, mx)
        s_txt, s_u = gen(C["SOLAR_URL"], C["SOLAR_MODEL"], [{"role": "user", "content": q}],
                         float(C.get("SOLAR_TEMP", .7)), float(C.get("SOLAR_TOP_P", .95)), seed, mx)
        cost["motif"].append(m_u); cost["solar"].append(s_u)
        motif_is_ga = rng.random() < 0.5           # per-item side randomization
        ga, na = (m_txt, s_txt) if motif_is_ga else (s_txt, m_txt)
        key.append({"n": n, "id": p.get("id"), "category": p.get("category"),
                    "ga": "motif" if motif_is_ga else "solar",
                    "na": "solar" if motif_is_ga else "motif"})
        md += [f"## {n}. [{p.get('category','-')}] {p.get('id','')}", "",
               f"**질문**\n\n{q}", "", "**답변 가**", "", "```", ga, "```", "",
               "**답변 나**", "", "```", na, "```", "",
               "판정: ", "", "---", ""]
        print(f"[blind] item {n}/{len(items)} generated", file=sys.stderr)

    (out / "blind_pack.md").write_text("\n".join(md), encoding="utf-8")
    (out / "blind_key.json").write_text(json.dumps(
        {"seed": seed, "items": key, "disclosed_cost": cost}, ensure_ascii=False, indent=2))
    print(f"[blind] pack  -> {out/'blind_pack.md'}   (give this to the reviewer)")
    print(f"[blind] key   -> {out/'blind_key.json'}  (do NOT open until scored)")

def score(a):
    C = os.environ
    out = Path(C["RESULTS_DIR"]) / "judge"
    pack = Path(a.pack) if a.pack else out / "blind_pack.md"
    keyf = Path(a.key) if a.key else out / "blind_key.json"
    K = json.loads(keyf.read_text())
    kmap = {i["n"]: i for i in K["items"]}

    text = pack.read_text(encoding="utf-8")
    verdicts = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"^##\s+(\d+)\.", line)
        if m: cur = int(m.group(1)); continue
        m = re.match(r"^판정:\s*(.+)$", line)
        if m and cur is not None:
            v = m.group(1).strip()
            if v: verdicts[cur] = v
    if not verdicts:
        sys.exit("[blind] no filled `판정:` lines found — score the pack first.")

    tally = dict(motif=0, solar=0, tie=0, unparsed=0)
    rows, scores = [], []
    for n, item in kmap.items():
        v = verdicts.get(n)
        if v is None: continue
        if v in ("가", "ga", "A"):      win = item["ga"]
        elif v in ("나", "na", "B"):    win = item["na"]
        elif v in ("무승부", "tie", "TIE"): win = "tie"
        else: win = "unparsed"
        tally[win] = tally.get(win, 0) + 1
        scores.append(1 if win == "motif" else -1 if win == "solar" else 0)
        rows.append({"n": n, "id": item["id"], "category": item["category"],
                     "reviewer_said": v, "final": win})

    import numpy as np
    n = len(scores)
    arr = np.asarray(scores, float)
    rng = np.random.default_rng(int(C.get("SEED", 1234)))
    boot = arr[rng.integers(0, arr.size, size=(int(C.get("BOOTSTRAP_N", 2000)), arr.size))].mean(axis=1) \
        if n else np.array([0.0])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    marg = float(C.get("JUDGE_MARGIN", 0.0))
    vd = "motif" if lo > marg else "solar" if hi < -marg else "INCONCLUSIVE"

    human = a.reviewer and a.reviewer.lower() not in ("claude", "assistant", "model")
    summary = {
        "n": n, "motif_wins": tally["motif"], "solar_wins": tally["solar"],
        "semantic_ties": tally["tie"], "parse_fail": tally.get("unparsed", 0),
        "order_effect": 0, "judge_disagree": 0,
        "motif_winrate_overall": round(tally["motif"] / n, 3) if n else None,
        "solar_winrate_overall": round(tally["solar"] / n, 3) if n else None,
        "score_mean": float(arr.mean()) if n else None, "score_ci95": [lo, hi],
        "verdict": vd, "judges": [f"blind:{a.reviewer}"],
        "judge_position_bias": {"blind": "n/a (single-order blinded pack)"},
        "judge_agreement": None,
        "disclosed_cost": {k: {"mean_completion_tokens":
                               round(sum(x["completion_tokens"] or 0 for x in v) / len(v), 1) if v else None,
                               "mean_chars": round(sum(x["chars"] for x in v) / len(v), 1) if v else None}
                           for k, v in K.get("disclosed_cost", {}).items()},
        "blinded": True, "reviewer_is_human": bool(human),
        "publishable": bool(human) and n >= 20,
        "note": ("blinded single-reviewer pack; identity hidden and sides randomized. "
                 "Publishable only with a human reviewer and n>=20 — a single LLM reviewer "
                 "cannot supply inter-judge agreement, and the harness author is not neutral."),
    }
    (out / "pairwise.json").write_text(json.dumps({"summary": summary, "rows": rows},
                                                  ensure_ascii=False, indent=2))
    print("[blind] summary:", json.dumps(summary, ensure_ascii=False)[:400])
    print(f"[blind] -> {out/'pairwise.json'} (aggregate.py picks this up)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make");  m.add_argument("--prompts"); m.add_argument("--limit", type=int)
    s = sub.add_parser("score"); s.add_argument("--reviewer", required=True)
    s.add_argument("--pack"); s.add_argument("--key")
    a = ap.parse_args()
    (make if a.cmd == "make" else score)(a)
