#!/usr/bin/env python3
"""Roll three disclosed rounds (floor / mid / reference) into one series view.

The series is deliberately run and DISCLOSED one tier at a time. This tool never
invents a series winner: it reports each round as published, and only states a
series-level direction when the rounds AGREE. A single-round win stays a
single-round win.

  ROUNDS_DIR=cases/2026-07-motif3-vs-solar2/results python3 series.py
  -> series_report.md next to the rounds

Reads each round's report.json (written by aggregate.py) plus its run_manifest.json
so the disclosed cost travels with the score.
"""
import json, os, sys
from pathlib import Path

ROUNDS = [("floor", "Motif 2.3bpw ↔ Solar F-v2"),
          ("mid", "Motif 4.5bpw ↔ Solar Q-v3"),
          ("reference", "Motif 8bit ↔ Solar T")]

R = Path(os.environ.get("ROUNDS_DIR", "results"))

def load(round_name):
    d = R / round_name
    rep, man = d / "report.json", d / "run_manifest.json"
    if not rep.exists():
        return None
    j = json.load(open(rep))
    m = json.load(open(man)) if man.exists() else {}
    return {"round": round_name, "publishable": j.get("publishable"),
            "integrity": (j.get("integrity") or {}).get("verdict"),
            "rows": j.get("rows", []), "run_id": (j.get("integrity") or {}).get("run_id"),
            "disclosure": m.get("disclosure", {})}

def direction(rows):
    """-> (motif_wins, solar_wins, inconclusive) over per-task verdicts."""
    v = [r[-1] for r in rows]
    return v.count("motif"), v.count("solar"), v.count("INCONCLUSIVE")

def main():
    got = [(name, label, load(name)) for name, label in ROUNDS]
    md = ["# Series — Motif-3-Beta vs Solar Open 2", "",
          "Three rounds, each **disclosed when it completed**, in the order run.",
          "No round is withheld; no single round is 'the' result.", "",
          "| round | pairing | status | Motif | Solar | inconclusive |",
          "|---|---|---|---|---|---|"]
    dirs = []
    for name, label, d in got:
        if d is None:
            md.append(f"| {name} | {label} | ⏳ not run yet | — | — | — |")
            continue
        mw, sw, inc = direction(d["rows"])
        dirs.append((name, mw, sw))
        flag = "✅ disclosed" if d["publishable"] else f"⛔ {d['integrity']} (not publishable)"
        md.append(f"| {name} | {label} | {flag} | {mw} | {sw} | {inc} |")

    # disclosed cost per round — a cheaper win is a different claim than a costlier one
    md += ["", "## Disclosed cost per round", "",
           "| round | thinking | motif build | solar build | motif runtime | solar runtime |",
           "|---|---|---|---|---|---|"]
    for name, _, d in got:
        if d is None: continue
        dc = d["disclosure"] or {}
        b = dc.get("builds", {}); rt = dc.get("runtime", {})
        f = lambda x: (x or {}).get("commit", "?")[:9] + ("(dirty)" if (x or {}).get("dirty") else "")
        md.append(f"| {name} | {dc.get('thinking')} | `{(b.get('motif') or '?').split('/')[-1]}` | "
                  f"`{(b.get('solar') or '?').split('/')[-1]}` | {f(rt.get('motif_fork'))} | {f(rt.get('solar_fork'))} |")

    # series-level statement — only when rounds agree
    md += ["", "## Series statement", ""]
    done = [d for d in dirs]
    if len(done) < 2:
        md.append("- Fewer than two rounds disclosed — **no series-level statement**.")
    else:
        leans = ["motif" if mw > sw else "solar" if sw > mw else "tie" for _, mw, sw in done]
        uniq = set(leans) - {"tie"}
        if len(uniq) == 1 and leans.count(list(uniq)[0]) >= 2:
            who = list(uniq)[0]
            md.append(f"- Rounds disclosed so far **agree**: they lean **{who}**. "
                      f"Stated as a direction across tiers, not a margin.")
        else:
            md.append("- Rounds **disagree or tie** — the honest series statement is that the "
                      "outcome is **tier-dependent**. Report each round on its own; do not "
                      "average them into a single winner.")
    md.append("")
    md.append("> `KL(own 8-bit ‖ build)` is a per-model cost and is never ranked across models. "
              "Where a side has no working 8-bit anchor, its KL cell reads MISSING, not zero.")

    out = R / "series_report.md"
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n[series] wrote {out}")

if __name__ == "__main__":
    main()
