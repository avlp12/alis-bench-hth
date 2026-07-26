#!/usr/bin/env python3
"""Aggregate the head-to-head into a report — publication-grade version.

Fixes from adversarial review:
  * EXPLICIT task->metric schema (no "first float" fallback -> no n_samples-as-score).
  * PAIRED bootstrap CI on per-item differences (lm-eval --log_samples) with an
    equivalence margin -> per-task verdict is motif / solar / INCONCLUSIVE.
  * INTEGRITY-gated: reads run_manifest.json; refuses a COMPLETE verdict unless
    every required cell is present and stamped with the current run_id.
  * KL reported per-model and explicitly labelled NON-comparable across models.

Outputs: report.md, report.csv, report.json in RESULTS_DIR.
"""
import csv, json, glob, os, re
from pathlib import Path
import numpy as np

R = Path(os.environ.get("RESULTS_DIR", "."))
TIER = os.environ.get("BPW_TIER", "?")
MARGIN = float(os.environ.get("EQUIV_MARGIN", "0.0"))      # practical-effect threshold (fraction)
BOOT = int(os.environ.get("BOOTSTRAP_N", "2000"))
SEED = int(os.environ.get("SEED", "1234"))

# ---- explicit metric schema: ordered acceptable "metric,filter" keys per task ----
METRIC_SCHEMA = {
    "mmlu_pro":           ["exact_match,custom-extract", "exact_match,none"],
    "bbh_cot_zeroshot":   ["exact_match,none", "exact_match,get-answer"],
    # gsm8k zero-shot: flexible-extract first. strict-match wants the exact
    # sentence form; flexible takes the last number and is robust to a model that
    # solves the problem while phrasing the conclusion its own way.
    "gsm8k_cot_zeroshot": ["exact_match,flexible-extract", "exact_match,strict-match"],
    # minerva exact_match requires the Minerva incantation ("Final Answer: The
    # final answer is X. I hope it is correct.") that only its 4-shot exemplars
    # taught. math_verify checks mathematical equivalence and is format-free.
    "minerva_math":       ["math_verify,none", "exact_match,none"],
    "ifeval":             ["prompt_level_strict_acc,none", "inst_level_strict_acc,none"],
}
GROUPS = {"bbh_cot_zeroshot": "bbh"}  # group tasks whose samples land in per-subtask files

def load_results(name):
    """task -> metrics dict, from lm-eval results*.json.

    Merges the `groups` block too: mmlu_pro / bbh / minerva_math are GROUPS, and
    depending on lm-eval version their aggregate row can live under `groups`
    rather than `results` — reading only `results` would silently drop them.
    """
    out = {}
    for f in glob.glob(str(R / "lm_eval" / name / "**" / "results*.json"), recursive=True):
        try: j = json.load(open(f))
        except Exception: continue
        for block in ("results", "groups"):
            for task, m in (j.get(block) or {}).items():
                if isinstance(m, dict):
                    out.setdefault(task, {}).update(m)
    return out

def primary(task, metrics):
    """(value, se, key) using the schema — NO generic fallback."""
    for key in METRIC_SCHEMA.get(task, []):
        if key in metrics:
            se = metrics.get(key.split(",")[0] + "_stderr,none")
            return metrics[key], se, key
    return None, None, None  # schema drift -> excluded from verdict, flagged loudly

def failure_rates(name):
    """(empty_content, extraction_miss, total) per model.

    The previous heuristic (>6000 chars, no think marker) was inverted for BOTH
    architectures: lm-eval stores only `message.content`, so a model capped
    mid-thinking yields EMPTY content and was never counted, while a model that
    reasons visibly and finishes normally trips the length test and raised a
    permanent false alarm. Measure the two things that actually mean "this item
    produced no usable answer":
      empty_content   -> the budget died before any answer (truncation)
      extraction_miss -> the filter could not find an answer in what was produced
    Both are symmetric across models and directly bound how much of a score is
    capability versus plumbing.
    """
    import glob as _g
    tot = empty = miss = 0
    for f in _g.glob(str(R / "lm_eval" / name / "**" / "samples_*.jsonl"), recursive=True):
        try:
            for line in open(f):
                r = json.loads(line); tot += 1
                resps = r.get("resps") or []
                txt = " ".join(x if isinstance(x, str) else str(x) for x in
                               (resps[0] if resps and isinstance(resps[0], list) else resps))
                if not txt.strip():
                    empty += 1; continue
                fr = r.get("filtered_resps")
                flat = " ".join(str(x) for x in (fr if isinstance(fr, list) else [fr]))
                if (not flat.strip()) or "[invalid]" in flat:
                    miss += 1
        except Exception:
            continue
    return empty, miss, tot

def load_samples(name, task, base_metric, filt):
    """{(subtask,doc_id): score} from lm-eval --log_samples jsonl.

    MUST filter on the sample's `filter` field: a task with several filters
    (e.g. gsm8k_cot strict-match AND flexible-extract) writes one record per
    (doc, filter), all carrying the same bare metric key. Keying only on
    (subtask, doc_id) lets one filter silently overwrite the other, so the
    paired diff would mix filters and report a number for the wrong metric.
    """
    pat = f"samples_{GROUPS.get(task, task)}*"
    d, seen_filters = {}, set()
    for f in glob.glob(str(R / "lm_eval" / name / "**" / pat), recursive=True):
        st = re.sub(r"^samples_|_\d{4}-\d.*$|\.jsonl$", "", os.path.basename(f))
        try:
            for line in open(f):
                r = json.loads(line)
                rf = r.get("filter")
                if rf is not None:
                    seen_filters.add(rf)
                    if filt and rf != filt:
                        continue          # wrong filter -> not our metric
                v = r.get(base_metric)
                if v is None:
                    continue
                d[(st, r.get("doc_id"))] = float(v)
        except Exception:
            continue
    return d, seen_filters

def bootstrap_paired(diffs):
    """(mean, lo95, hi95, p) of the paired mean difference; p = 2-sided bootstrap."""
    a = np.asarray(diffs, float)
    if a.size == 0: return (None, None, None, None)
    rng = np.random.default_rng(SEED)
    means = a[rng.integers(0, a.size, size=(BOOT, a.size))].mean(axis=1)
    # 2-sided bootstrap p: how often the resampled mean sits on the other side of 0
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return (float(a.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), float(min(1.0, p)))

def holm(pvals, alpha=0.05):
    """Holm-Bonferroni: -> {index: significant?}. Controls family-wise error
    across the task family (5 tasks = 5 tests; uncorrected -> inflated wins)."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    order = sorted(idx, key=lambda i: pvals[i])
    m, sig, rejected = len(order), {}, True
    for rank, i in enumerate(order):
        thresh = alpha / (m - rank)
        if rejected and pvals[i] <= thresh:
            sig[i] = True
        else:
            rejected = False; sig[i] = False
    return sig

def verdict(lo, hi, sig):
    """Directional call requires BOTH the CI to clear the equivalence margin AND
    Holm-corrected significance."""
    if lo is None: return "no-paired"
    if not sig:    return "INCONCLUSIVE"
    if lo > MARGIN:  return "motif"
    if hi < -MARGIN: return "solar"
    return "INCONCLUSIVE"

def load_integrity():
    man = R / "run_manifest.json"
    if not man.exists():
        return {"verdict": "NO-MANIFEST", "suite_mode": "?", "run_id": "?"}
    j = json.loads(man.read_text())
    return {"verdict": j.get("integrity", {}).get("verdict", "UNCHECKED"),
            "suite_mode": j.get("suite_mode", "?"), "run_id": j.get("run_id", "?"),
            "detail": j.get("integrity", {})}

def load_hret(name):
    p = R / "hret" / name / "_summary.json"
    if not p.exists(): return {}
    try: j = json.load(open(p))
    except Exception: return {}
    def _score(d):
        if isinstance(d, (int, float)): return d
        if not isinstance(d, dict): return None
        # HRET's EvaluationResult.to_dict() nests everything under "metrics";
        # checking only the top level renders every Korean row as "unresolved".
        if isinstance(d.get("metrics"), dict): d = d["metrics"]
        for k in ("accuracy", "score", "acc", "exact_match"):
            if k in d and isinstance(d[k], (int, float)): return d[k]
        return None
    out = {}
    for ds, v in j.items():
        if isinstance(v, dict) and ("raw" in v or "penalized" in v):
            # dual-pass shape: raw correctness is primary, penalty reported apart
            if "raw" in v:       out[f"hret:{ds}"] = _score(v["raw"])
            if "penalized" in v: out[f"hret:{ds} (lang-penalized)"] = _score(v["penalized"])
        else:
            out[f"hret:{ds}"] = _score(v)
    return out

def load_kl(name):
    p = R / "kl" / f"{name}_kl.txt"
    if not p.exists(): return None
    m = re.search(r"KL\(ref\|\|cand\)=([0-9.]+)", p.read_text())
    return float(m.group(1)) if m else None

def load_thru(name):
    p = R / "throughput" / f"{name}.json"
    return json.load(open(p)) if p.exists() else {}

def main():
    R.mkdir(parents=True, exist_ok=True)
    integ = load_integrity()
    publishable = integ["verdict"] == "COMPLETE"
    agg = {n: load_results(n) for n in ("motif", "solar")}

    md = [f"# Motif-3-Beta vs Solar Open 2 — head-to-head @ {TIER}", ""]
    banner = ("✅ COMPLETE — all cells present & fresh" if publishable else
              f"⛔ **{integ['verdict']}** — NOT publishable; per-task rows shown for debugging only")
    md += [f"**Integrity:** {banner}  ·  run_id `{integ['run_id']}`  ·  suite_mode `{integ['suite_mode']}`",
           f"**Verdict rule:** paired bootstrap (N={BOOT}), equivalence margin ±{MARGIN}; CI crossing the margin ⇒ INCONCLUSIVE.", ""]
    if integ.get("detail", {}).get("missing") or integ.get("detail", {}).get("stale"):
        md += ["> missing: " + ", ".join(integ["detail"].get("missing", []) or ["—"]),
               "> stale: " + ", ".join(integ["detail"].get("stale", []) or ["—"]), ""]

    # ---- lm-eval tasks: paired ----
    # pass 1: collect paired stats
    stats, schema_drift, unpaired = [], [], []
    for task in METRIC_SCHEMA:
        mv, mse, mk = primary(task, agg["motif"].get(task, {}))
        sv, sse, sk = primary(task, agg["solar"].get(task, {}))
        if mv is None or sv is None:
            schema_drift.append(task); continue
        base, filt = mk.split(",")[0], (mk.split(",")[1] if "," in mk else None)
        ms, fm = load_samples("motif", task, base, filt)
        ss, fs = load_samples("solar", task, base, filt)
        shared = sorted(set(ms) & set(ss))
        if not shared:
            unpaired.append(f"{task} (motif n={len(ms)}, solar n={len(ss)}; filters seen: "
                            f"{sorted(fm | fs) or 'none'}; wanted '{filt}')")
        dmean, lo, hi, p = bootstrap_paired([ms[k] - ss[k] for k in shared])
        stats.append(dict(task=task, base=base, filt=filt, mv=mv, mse=mse, sv=sv, sse=sse,
                          dmean=dmean, lo=lo, hi=hi, p=p, n=len(shared)))

    # Korean suite: same paired treatment as English (per-item, bootstrap, Holm)
    def _ko_items(name, ds):
        f = R / "hret" / name / f"{ds}_items.json"
        if not f.exists(): return {}
        try: return {r["key"]: r["score"] for r in json.load(open(f))}
        except Exception: return {}
    ko_ds = sorted({p.name[:-len("_items.json")]
                    for n in ("motif", "solar")
                    for p in (R / "hret" / n).glob("*_items.json")} ) if (R / "hret").exists() else []
    for ds in ko_ds:
        mi, si = _ko_items("motif", ds), _ko_items("solar", ds)
        shared = sorted(set(mi) & set(si))
        if not shared:
            unpaired.append(f"hret:{ds} (motif n={len(mi)}, solar n={len(si)}; no shared item keys)")
        dmean, lo, hi, p_ = bootstrap_paired([mi[k] - si[k] for k in shared])
        stats.append(dict(task=f"hret:{ds}", base="accuracy", filt=None,
                          mv=sum(mi.values())/len(mi) if mi else 0.0, mse=None,
                          sv=sum(si.values())/len(si) if si else 0.0, sse=None,
                          dmean=dmean, lo=lo, hi=hi, p=p_, n=len(shared)))

    # pass 2: Holm correction across the task family, then verdicts
    sig = holm([s["p"] for s in stats])
    md += ["## Scored tasks — paired (Holm-corrected across the family)", "",
           "| task | metric[filter] | Motif | Solar | Δ mean [95% CI] | p | verdict |",
           "|---|---|---|---|---|---|---|"]
    rows = []
    for i, s in enumerate(stats):
        vd = verdict(s["lo"], s["hi"], sig.get(i, False))
        ci = "—" if s["lo"] is None else f"{s['dmean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] (n={s['n']})"
        pp = "—" if s["p"] is None else f"{s['p']:.3f}"
        md.append(f"| {s['task']} | {s['base']}[{s['filt'] or '-'}] | {s['mv']:.3f}±{s['mse'] or 0:.3f} | "
                  f"{s['sv']:.3f}±{s['sse'] or 0:.3f} | {ci} | {pp} | {vd} |")
        rows.append([s["task"], f"{s['base']},{s['filt']}", s["mv"], s["mse"], s["sv"], s["sse"],
                     s["dmean"], s["lo"], s["hi"], s["n"], s["p"], vd])

    if schema_drift:
        md += ["", f"> ⚠ **schema drift (excluded):** {', '.join(schema_drift)} — expected metric key absent; "
               "fix METRIC_SCHEMA or the task config, do NOT guess a metric."]
    if unpaired:
        md += ["", "> 🛑 **NO PAIRED SAMPLES — the statistical layer is OFF for these tasks.** "
               "Re-run lm-eval with `--log_samples` (and check the filter name); marginal means alone "
               "CANNOT decide a winner:"] + [f">   - {u}" for u in unpaired]

    # ---- HRET: marginal only (no per-item paired test available) ----
    hm, hs = load_hret("motif"), load_hret("solar")
    if hm or hs:
        md += ["", "## Korean suite (HRET) — marginal (no paired test; treat Δ cautiously)", "",
               "| dataset | Motif | Solar | note |", "|---|---|---|---|"]
        for ds in sorted(set(hm) | set(hs)):
            a, b = hm.get(ds), hs.get(ds)
            note = "metric unresolved" if (a is None or b is None) else ""
            md.append(f"| {ds} | {a if a is not None else '—'} | {b if b is not None else '—'} | {note} |")

    # ---- judge ----
    jp = R / "judge" / "pairwise.json"
    if jp.exists():
        js = json.load(open(jp)).get("summary", {})
        md += ["", "## Korean open-gen (pairwise, neutral judges)", "",
               f"- decided: Motif **{js.get('motif_wins')}** / Solar **{js.get('solar_wins')}** · "
               f"ties {js.get('semantic_ties','?')} · parse-fail {js.get('parse_fail','?')} · "
               f"order-effect {js.get('order_effect','?')} (n={js.get('n')})",
               f"- Motif win-rate **over all n**: {js.get('motif_winrate_overall')}  "
               f"(NOT the misleading 'vs decided'); judges: {js.get('judges')}",
               f"- judge agreement: {js.get('judge_agreement')} · verdict: **{js.get('verdict','?')}**"]

    # ---- truncation: silent score corruption for reasoning models ----
    md += ["", "## Plumbing failures (items that produced no usable answer)", "",
           "| model | empty content (budget died) | extraction miss | total |", "|---|---|---|---|"]
    plumb_bad = False
    for n in ("motif", "solar"):
        e, m, tot = failure_rates(n)
        if tot and (e + m) / tot > 0.02: plumb_bad = True
        md.append(f"| {n} | {e} ({e/tot:.1%}) | {m} ({m/tot:.1%}) | {tot} |" if tot
                  else f"| {n} | — | — | 0 |")
    if plumb_bad:
        md += ["", "> 🛑 **Above 2% these scores mix capability with plumbing.** Empty content means "
               "the token budget died before an answer; an extraction miss means the filter could not "
               "find one in what was produced. Neither is evidence about the model. Fix and re-run."]

    # ---- disclosed cost ledger: what each model SPENT to reach its peak ----
    disc = (json.loads((R / "run_manifest.json").read_text()).get("disclosure")
            if (R / "run_manifest.json").exists() else None)
    md += ["", "## Disclosed cost — the price of 'peak' (best-vs-best is only honest with this)", ""]
    if disc:
        s = disc.get("sampling", {}); b = disc.get("builds", {})
        md += [f"- thinking: `{disc.get('thinking')}` · max_tokens: `{disc.get('max_tokens')}` · seed: `{disc.get('seed')}`",
               f"- sampling — motif temp={s.get('motif',{}).get('temp')}/top_p={s.get('motif',{}).get('top_p')} · "
               f"solar temp={s.get('solar',{}).get('temp')}/top_p={s.get('solar',{}).get('top_p')}",
               f"- builds — motif `{b.get('motif')}` · solar `{b.get('solar')}`",
               f"- HRET language penalty mode: `{disc.get('hret_penalize')}`"]
        for who, rt in (disc.get("runtime") or {}).items():
            if rt and rt.get("path"):
                dirty = "⚠ DIRTY tree — no commit reproduces this runtime" if rt.get("dirty") else "clean"
                md.append(f"- runtime {who}: `{rt.get('commit') or '?'}` ({dirty}, {rt.get('dirty_files', 0)} modified)")
    else:
        md.append("- ⚠ no manifest — peak settings and runtime provenance were NOT captured.")
    if jp.exists():
        dc = json.load(open(jp)).get("summary", {}).get("disclosed_cost", {})
        if dc:
            md += ["", "| model | mean completion tokens | mean reasoning tokens | mean chars |", "|---|---|---|---|"]
            for k in ("motif", "solar"):
                c = dc.get(k, {})
                md.append(f"| {k} | {c.get('mean_completion_tokens')} | {c.get('mean_reasoning_tokens')} | {c.get('mean_chars')} |")
            md.append("")
            md.append("> A win bought with materially more tokens is a *costlier* win — read this table beside the verdicts.")

    # ---- KL (per-model; NOT cross-model comparable) ----
    md += ["", "## Quantization cost — KL(own 8-bit ‖ build). ⚠ per-model only, do NOT rank across models", ""]
    for n in ("motif", "solar"):
        kl = load_kl(n); md.append(f"- {n}: {'—' if kl is None else f'{kl:.5f}'}")

    # ---- efficiency ----
    tm, ts = load_thru("motif"), load_thru("solar")
    if tm or ts:
        md += ["", "## Efficiency (same box) — report both; chars/s ≠ info, tok/s ≠ tokenizer-neutral", "",
               "| lang | metric | Motif | Solar |", "|---|---|---|---|"]
        for lang in ("KO", "EN", "code"):
            for metric in ("chars_s", "decode_tok_s", "ttft_s"):
                md.append(f"| {lang} | {metric} | {(tm.get(lang) or {}).get(metric)} | {(ts.get(lang) or {}).get(metric)} |")

    # ---- overall ----
    decisive = [r for r in rows if r[-1] in ("motif", "solar")]
    md += ["", "## Overall"]
    if not publishable:
        md.append("- ⛔ Integrity not COMPLETE — **no overall verdict issued.**")
    elif not rows:
        md.append("- no scored tasks.")
    else:
        m_w = sum(r[-1] == "motif" for r in rows); s_w = sum(r[-1] == "solar" for r in rows)
        inc = sum(r[-1] == "INCONCLUSIVE" for r in rows)
        md.append(f"- decisive tasks — Motif {m_w} / Solar {s_w} / inconclusive {inc} (of {len(rows)}).")
        md.append("- Report per-task; a single headline number is not issued (capability × cost, see KL/efficiency).")

    (R / "report.md").write_text("\n".join(md))
    with open(R / "report.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "metric_filter", "motif", "motif_se", "solar", "solar_se",
                                       "delta_mean", "ci_lo", "ci_hi", "n_paired", "p_boot", "verdict"])
        w.writerows(rows)
    (R / "report.json").write_text(json.dumps(
        {"integrity": integ, "tier": TIER, "margin": MARGIN, "rows": rows,
         "schema_drift": schema_drift, "publishable": publishable}, indent=2, ensure_ascii=False))
    print("\n".join(md)); print(f"\n[agg] wrote report.md / .csv / .json to {R}")

if __name__ == "__main__":
    main()
