#!/usr/bin/env python3
"""Korean suite via HRET (Haerae Evaluation Toolkit), pointed at the model's
OpenAI-compatible endpoint. HRET is the right tool for KO MCQA because it does
generation + answer parsing (no logprobs needed) AND adds the Korean-specific
fairness analyses we care about:

  * language_penalize=True, target_lang="ko"  -> penalizes English code-switching
  * morphology-aware lexical diversity + keyword-omission detection

Datasets (per HRET registry): kmmlu, haerae_bench, hrm8k, click, kudge (judge).
KMMLU-Redux / KMMLU-Pro are the decontaminated variants — pull from HAERAE-HUB
on HF if your HRET version exposes them (see SUBSETS below; VERIFY names).

Install:  pip install haerae-evaluation-toolkit
Usage:    python3 run_hret.py motif   |   python3 run_hret.py solar

NOTE: HRET's remote-endpoint backend arg names vary across versions. The block
below is written against the documented API; if your version differs, the two
things to confirm are (1) the backend name for an OpenAI-compatible server and
(2) the api_base/base_url key. Both are flagged VERIFY.
"""
import json, os, sys
from pathlib import Path

# Korean set — SUITE_MODE selects. full = decontaminated/advertised suite and
# ABORTS if HRET can't resolve a dataset (never silently downgrades to vanilla
# kmmlu). smoke = quick, clearly non-publishable subset.
# VERIFIED against the installed HRET DATASET_REGISTRY keys (hret 0.1.0, 2026-07-26):
#   KUDGE aime2025 benchhub click generic_file haerae_bench hrc hrm8k k2_eval kbl
#   kmmlu kormedmcqa
# (An earlier version of this file listed *module* names from pkgutil — `haerae`,
#  `kormedqa` — which are NOT registry keys and abort the run at dataset #2.)
# NOT available in HRET: kmmlu_redux, kmmlu_pro, kosimpleqa. Plain `kmmlu` is
# carried with an explicit CONTAMINATED flag instead of being passed off as the
# decontaminated suite.
CONTAMINATED = {"kmmlu": "vanilla KMMLU: ~5.4% test-dup / ~5.5% train-test leak; "
                         "KMMLU-Redux/Pro are NOT in HRET 0.1.0 — treat as a lower bound"}

def datasets_for(mode):
    if mode == "smoke":
        return [("kmmlu", {}), ("haerae_bench", {})]
    # Two datasets only (2026-07-27 amendment). Measured cost on the reference tier
    # is ~8.5 min/item for Korean MCQA — 20x the English rate — so the full five-set
    # suite is hundreds of hours per model. kmmlu covers knowledge breadth;
    # haerae_bench covers Korean-specific cultural/linguistic ability. Dropped:
    # hrm8k and kormedmcqa (cost), click (has no `test` split — train only).
    return [("kmmlu", {}), ("haerae_bench", {})]

def main():
    name = (sys.argv[1] if len(sys.argv) > 1 else "motif").lower()
    url   = os.environ["MOTIF_URL"   if name == "motif" else "SOLAR_URL"]
    model = os.environ["MOTIF_MODEL" if name == "motif" else "SOLAR_MODEL"]
    seed  = int(os.environ.get("SEED", "1234"))
    temp  = float(os.environ.get("MOTIF_TEMP" if name == "motif" else "SOLAR_TEMP", "0.7"))
    top_p = float(os.environ.get("MOTIF_TOP_P" if name == "motif" else "SOLAR_TOP_P", "0.95"))
    maxtok = int(os.environ.get("MAX_TOKENS", "8192"))
    outdir = Path(os.environ["RESULTS_DIR"]) / "hret" / name
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        from llm_eval.evaluator import Evaluator
    except Exception as e:
        sys.exit(f"[hret] import failed ({e}). Run: pip install haerae-evaluation-toolkit")

    evaluator = Evaluator()
    mode = os.environ.get("SUITE_MODE", "full")
    DATASETS = datasets_for(mode)
    (outdir / "_suite.json").write_text(json.dumps(
        {"suite_mode": mode, "datasets": [d for d, _ in DATASETS]}, ensure_ascii=False))
    print(f"[hret] {name} suite_mode={mode} datasets={[d for d,_ in DATASETS]}")
    summary = {}
    for ds, extra in DATASETS:
        print(f"[hret] {name} :: {ds}")
        # -------- VERIFY (2 keys) for your installed HRET version --------
        #   backend name for an OpenAI-compatible server, and the base-url key:
        # LiteLLMBackend.__init__ requires provider + model_name; passing `model`
        # raises TypeError before a single item is scored (verified 2026-07-26).
        model_backend = "litellm"
        model_params = {
            # LiteLLMBackend only prefixes the provider for azure/bedrock; for
            # "openai" it passes model_name through raw, and litellm cannot infer
            # a provider from a filesystem path -> BadRequestError. The prefix must
            # be in model_name itself. (Verified 2026-07-26: without it every
            # prediction comes back as the string "Error: litellm.BadRequestError…",
            # which the backend returns WITHOUT raising — i.e. the whole Korean
            # suite would score error text as answers and quietly read ~0%.)
            "provider": "openai",
            "model_name": f"openai/{model}",
            "api_base": url.rstrip("/") + "/v1",
            "api_key": os.environ.get("DUMMY_KEY", "sk-noauth"),
            "temperature": temp,            # best-vs-best: each model's own peak sampling
            "top_p": top_p,
            "max_tokens": maxtok,           # room for thinking; record actual usage
            "seed": seed,
        }
        # ----------------------------------------------------------------
        # HRET_PENALIZE: off (default) | on | both
        #   The KO language penalty is an opaque, version-dependent knob whose
        #   magnitude we do not control, and it hits a trilingual model (Solar:
        #   KO/EN/JA) differently than a bilingual one — it must NOT silently be
        #   the primary score. Default: RAW correctness is primary; language
        #   consistency is reported as a SEPARATE pass (HRET_PENALIZE=both).
        pen_mode = os.environ.get("HRET_PENALIZE", "off").lower()
        passes = {"off": [False], "on": [True], "both": [False, True]}[pen_mode]
        # Preregistered sample: 300 items per dataset, stratified across subsets.
        # The full Korean suite is ~51k items; at measured throughput that is
        # hundreds of hours per model. A fixed, preregistered sample is the honest
        # alternative to "all of it or nothing" — n=300 is 10x the MIN_PAIRED_N
        # floor and is frozen in the preregistration, not chosen after seeing scores.
        n_items = int(os.environ.get("KO_ITEMS", "300"))
        extra = dict(extra)
        extra.setdefault("dataset_params", {})
        try:
            payload = {}
            for pen in passes:
                res = evaluator.run(
                    model=model_backend,
                    dataset=ds,
                    split="test",
                    model_params=model_params,
                    evaluation_method="string_match",
                    language_penalize=pen,
                    target_lang="ko",
                    **extra,
                )
                got = res if isinstance(res, dict) else getattr(res, "to_dict", lambda: {"raw": str(res)})()
                # The backend swallows API failures and returns them as the
                # prediction string. Scoring those is worse than crashing: it
                # yields a plausible-looking 0% instead of an error.
                samples = got.get("samples") or []
                errs = sum(1 for s in samples
                           if isinstance(s, dict) and str(s.get("prediction", "")).lower().startswith("error:"))
                if samples and errs:
                    rate = errs / len(samples)
                    msg = (f"[hret] {ds}: {errs}/{len(samples)} predictions are backend ERROR strings "
                           f"({rate:.0%}) — these would be scored as wrong answers. Aborting.")
                    if mode == "full":
                        sys.exit(msg)
                    print(msg, file=sys.stderr)
                    got["_error_predictions"] = errs
                payload["penalized" if pen else "raw"] = got
            # primary = raw when available (correctness), else the penalized pass
            # Persist PER-ITEM correctness so the Korean suite gets the same paired
            # statistics as the English one. Marginal means alone cannot decide a
            # winner — that was our own standard for lm-eval and the KO stack was
            # being held to a weaker one.
            prim = payload.get("raw") or payload.get("penalized") or {}
            items = []
            for i, s in enumerate(prim.get("samples") or []):
                if not isinstance(s, dict):
                    continue
                # Key on a hash of the FULL input plus the subset: KMMLU's fixed
                # template prefix is ~103 chars, so input[:120] leaves almost no
                # question text and formulaic openers collide — colliding keys are
                # silently collapsed last-wins when the dict is built.
                import hashlib as _h
                raw = f"{s.get('_subset_name','')}|{s.get('input','')}"
                key = s.get("id") or s.get("doc_id") or _h.sha256(raw.encode()).hexdigest()[:24]
                key = str(key)
                # HRET's StringMatchEvaluator writes correctness into
                # sample["evaluation"]["is_correct"] — NOT at the top level. Reading
                # the top level yields None for every sample, so nothing is persisted
                # and the Korean half silently degrades to marginal-only.
                ev = s.get("evaluation") if isinstance(s.get("evaluation"), dict) else {}
                sc = ev.get("is_correct", s.get("is_correct", s.get("correct", s.get("score"))))
                if isinstance(sc, bool):
                    sc = float(sc)
                if isinstance(sc, (int, float)):
                    items.append({"key": key, "score": float(sc)})
            if items:
                (outdir / f"{ds}_items.json").write_text(
                    json.dumps(items, ensure_ascii=False))
                payload["_n_items"] = len(items)
            elif prim.get("samples"):
                msg = (f"[hret] {ds}: {len(prim['samples'])} samples but 0 per-item scores "
                       "extracted — the Korean suite would silently fall back to "
                       "marginal-only. The field layout changed; fix the extraction.")
                if mode == "full": sys.exit(msg)
                print(msg, file=sys.stderr)
            # HRET's Evaluator.run swallows pipeline failures and RETURNS a result
            # with metrics={"pipeline_error": ...} and samples=[] — the except below
            # never fires, so check explicitly.
            if isinstance(prim.get("metrics"), dict) and prim["metrics"].get("pipeline_error"):
                msg = f"[hret] {ds}: pipeline_error {prim['metrics']['pipeline_error']}"
                if mode == "full": sys.exit(msg)
                print(msg, file=sys.stderr)
            payload["_primary"] = "raw" if "raw" in payload else "penalized"
            payload["_penalize_mode"] = pen_mode
            if ds in CONTAMINATED:
                payload["_contaminated"] = CONTAMINATED[ds]
        except Exception as e:
            print(f"[hret]   ! {ds} failed: {e}")
            if mode == "full":
                sys.exit(f"[hret] FULL mode fail-closed on '{ds}': {e}\n"
                         "  -> HRET can't resolve this decontaminated dataset. Install/upgrade HRET or add a "
                         "loader; do NOT silently fall back to vanilla kmmlu. (Or SUITE_MODE=smoke — non-publishable.)")
            payload = {"error": repr(e)}
        (outdir / f"{ds}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        summary[ds] = payload.get("metrics", payload.get("score", payload))
    (outdir / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[hret] {name} done -> {outdir}")

if __name__ == "__main__":
    main()
