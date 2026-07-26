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

### Series structure — THREE ROUNDS, all reported

This matchup runs **one tier at a time, and each round is DISCLOSED as soon as it
finishes** — results are not held back until the series ends:

| round | pairing | runs when |
|---|---|---|
| **R1 floor** | Motif 2.3bpw ↔ Solar F-v2 | first (both smallest; smoke + real result) |
| **R2 mid** | Motif 4.5bpw ↔ Solar Q-v3 | after R1 |
| **R3 reference** | Motif 8bit ↔ Solar T | last (needs the Motif 8bit rebuild) |

**Binding disclosure rule.** Every round is published when it completes, whatever it
says — including rounds the Motif side loses. A round may not be withheld, quietly
delayed, or dropped from the writeup because the result is unfavourable, and no
single round may be presented as "the" result of the series.

Each round is disclosed as: the round's `report.md` + `run_manifest.json` (settings,
runtime commits, disclosed token cost) committed to `cases/2026-07-motif3-vs-solar2/
results/<round>/`, plus a short public note. Publishing a round **before** the next
one runs is deliberate: it prevents choosing which tier to report after seeing all
three. If a round cannot run, the reason is recorded here as a dated amendment
before the next round starts.

Each round gets its own `run_id`, manifest, and report under
`results/<round>/`; `series.py` rolls them up. Because three rounds means three
families of tests, per-round Holm correction stands and the **series-level claim
requires agreement across rounds** — a single-round win is reported as exactly that.

Tiers are paired by ROLE, not matched bpw — the protocol explicitly rejects
matched-recipe quantization:

| tier | Motif (alis-dwq / avlp12) | Solar (Kimi K3) | status |
|---|---|---|---|
| floor | 2.3bpw · 85 GB | F-v2 · 68 GB | first pairing to run |
| mid | 4.5bpw · 167 GB | Q-v3 · 146 GB | |
| reference (KL anchor) | 8bit · 334 GB | T · 266 GB | ⚠ Motif 8bit KNOWN BROKEN, rebuilding |

⚠ **Asymmetric KL anchor (2026-07-26).** The Motif 8-bit reference build is
defective (`mx.split` silent corruption above 2³¹ elements, built before the
workaround; reproduced at greedy, publicly flagged). Until it is rebuilt, Motif
has **no KL anchor**, so the quantization-cost cell must be reported as MISSING
for Motif — never as zero, and never compared against Solar's KL.

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

Pick ONE and record it here — the choice caps what may be claimed:

- [ ] **≥2 neutral judge APIs** (`judge_pairwise.py`), both A/B orders → publishable
- [ ] **blinded pack, human Korean-native reviewer** (`blind_pack.py`, n≥20) → publishable
- [ ] 1 API judge + a second model → marginal, state the limitation
- [ ] harness author's model alone → **NOT publishable** (authorship conflict + no
      agreement statistic + not blind to the side it built)

Ties, parse-failures and order-effects are counted separately; win-rate is over **all n**.
Judges/reviewer: ______  ·  Korean-native spot-check of ≥20 items: ______

If no judge is available, run everything else and mark open-generation **NOT RUN** —
MCQA, IFEval, math and code are all scored mechanically and need no judge.

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
