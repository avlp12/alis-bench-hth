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
# VERIFIED against the installed HRET registry (2026-07-26, hret 0.1.0):
#   aime2025 benchhub click haerae hrc hrm8k k2_eval kbl kmmlu kormedqa kudge
# NOT available in HRET: kmmlu_redux, kmmlu_pro, kosimpleqa. The protocol prefers
# the decontaminated KMMLU variants, so plain `kmmlu` is carried with an explicit
# CONTAMINATED flag rather than being passed off as the decontaminated suite.
CONTAMINATED = {"kmmlu": "vanilla KMMLU: ~5.4% test-dup / ~5.5% train-test leak; "
                         "KMMLU-Redux/Pro are NOT in HRET 0.1.0 — treat as a lower bound"}

def datasets_for(mode):
    if mode == "smoke":
        return [("kmmlu", {}), ("haerae", {})]
    return [("kmmlu", {}), ("haerae", {}), ("hrm8k", {}), ("click", {}), ("kormedqa", {})]

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
        model_backend = "litellm"          # or "openai" ; litellm passes through to any OpenAI-compatible server
        model_params = {
            "model": f"openai/{model}",     # litellm "openai/<name>" -> generic OpenAI-compatible route
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
                payload["penalized" if pen else "raw"] = got
            # primary = raw when available (correctness), else the penalized pass
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
