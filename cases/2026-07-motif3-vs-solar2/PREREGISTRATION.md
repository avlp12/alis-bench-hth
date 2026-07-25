# Pre-registration — Motif-3-Beta vs Solar Open 2

**Status:** DRAFT — becomes binding when both parties sign and the hash is recorded below.
**Rule:** this file is frozen *before* either side sees any benchmark result. Anything
changed afterwards must be added as a dated amendment, never edited in place.

| | |
|---|---|
| Matchup | `Motif-Technologies/Motif-3-Beta` (314B/13B MoE) vs `upstage/Solar-Open-2` (250B/15B MoE) |
| Serving | `mlx_lm.server` (OpenAI-compatible), Apple Silicon, one model per box |
| Quantization | **each side its own optimum** — Motif by alis-dwq (avlp12), Solar by Kimi K3 |
| Signatories | Motif side: avlp12 · Solar side: __(Kimi/operator)__ · Date: ______ |

---

## 1. Primary endpoint (decided before results)

**Primary metric:** per-task accuracy as defined by `METRIC_SCHEMA` in
`harness/aggregate.py` (`task → metric[filter]`, no fallback).

**Winner rule, per task:** paired bootstrap over per-item differences
(N=2000, seed 1234) → directional call requires **both**
(a) the 95% CI to clear the equivalence margin `EQUIV_MARGIN`, and
(b) Holm-corrected significance at α=0.05 across the task family.
Otherwise the task is **INCONCLUSIVE**. Overlapping marginal CIs are never a win.

**No single headline number.** The result is reported as per-task verdicts plus the
disclosed-cost ledger. Aggregating tasks into one score is explicitly out of scope.

**Equivalence margin:** `EQUIV_MARGIN = ______` (set before running; 0.0 means any
statistically-clear difference counts, which is permissive — consider 0.01–0.02).

## 2. Task suite (frozen)

Anything not listed is exploratory and may not be reported as a result.

| domain | tasks | harness |
|---|---|---|
| EN reasoning/math/IF | `mmlu_pro`, `bbh`, `gsm8k_cot`, `minerva_math`, `ifeval` | lm-eval 0.4.9, generative |
| KO knowledge/reasoning | `kmmlu`⚠, `haerae`, `hrm8k`, `click`, `kormedqa` | HRET 0.1.0 |
| KO open generation | LogicKor corpus (pinned hash: ______) | pairwise, ≥2 judges |

⚠ **Contamination disclosure:** HRET 0.1.0 does **not** ship KMMLU-Redux/Pro or
KoSimpleQA. Plain `kmmlu` is used and is automatically flagged `_contaminated`
(~5.4% test-dup, ~5.5% train–test leak). It is reported as a **lower bound**, and
is NOT presented as the decontaminated suite. Adding a decontaminated loader is
tracked as an improvement, not a silent substitution.

**Not automated here** (must not be claimed as run): GPQA-Diamond, LiveCodeBench,
HumanEval+/MBPP+, AIME/HMMT, RULER/long-context. Each requires its own harness.

## 3. Peak settings (each side declares its own, frozen here)

| | Motif | Solar |
|---|---|---|
| build | ______ | ______ |
| bpw / GB | ______ | ______ |
| 8-bit reference | ______ | ______ |
| thinking | ______ | ______ |
| temperature / top_p | ______ | ______ |
| max_tokens | ______ | ______ |
| runtime commit | `avlp12/mlx-lm@2a07e51` (clean) | ______ |

Self-consistency / best-of-k is **not implemented** in this harness; every item is a
single sample for both sides.

## 4. Judges (open generation)

≥2 independent families, neither a contender. Both A/B orders per item. Ties,
parse-failures and order-effects are counted separately; win-rate is over **all n**.
Judges: ______, ______. Korean-native spot-check of ≥20 items: ______.

## 5. Integrity

A verdict is emitted only when `manifest.py check` reports COMPLETE: every required
cell present and stamped with the current `run_id` (hash of models, endpoints,
config, harness code, suite mode). `ALLOW_PARTIAL=1` output is labelled PARTIAL and
is not a result.

## 6. What each side may claim

- **Track 1 (high-precision base)** → "which base model is better" — requires both at 8-bit/bf16.
- **Track 2 (matched-resource quant)** → "which quantization is better at equal budget".
- **Track 3 (best-submitted)** → "which shipped system is better", **always** quoted with
  bpw/GB and tokens spent.

`KL(own-8bit ‖ build)` is a per-model cost. **Cross-model KL ranking is prohibited.**

## 7. Amendments

| date | change | reason |
|---|---|---|
| | | |
