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
| KO knowledge/reasoning | `kmmlu`⚠, `haerae_bench`, `hrm8k`, `click`, `kormedmcqa` | HRET 0.1.0 |
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

**R3 (reference tier) — frozen 2026-07-26:**

| | Motif | Solar |
|---|---|---|
| build | Motif-3-Beta 8-bit (rebuilt) | Solar-Open-2 T (Q8) |
| shards / bytes | 77 / 334,382,459,970 | 62 / ~266 GB |
| served on | epsilon 10.0.0.2:8081 | gesicht 127.0.0.1:8082 |
| 8-bit reference | n/a — this build IS the anchor | n/a — same |
| adapter | none | none (T ships none) |
| thinking | on (template default) | on (`reasoning_effort=high`) |
| temperature / top_p | 1.0 / 0.95 | 1.0 / 1.0 |
| max_tokens | 32768 | 32768 |
| fewshot | 0 | 0 |
| runtime commit | `avlp12/mlx-lm@2a07e51` (clean) | kimi fork `@7937fda` (clean) |

Self-consistency / best-of-k is **not implemented** in this harness; every item is a
single sample for both sides.

## 4. Judges (open generation)

Pick ONE and record it here — the choice caps what may be claimed:

- [ ] **≥2 neutral judge APIs** (`judge_pairwise.py`), both A/B orders → publishable
- [ ] **blinded pack, human Korean-native reviewer** (`blind_pack.py`, n≥20) → publishable
- [ ] 1 API judge + a second model → marginal, state the limitation
- [ ] harness author's model alone → **NOT publishable** (authorship conflict + no
      agreement statistic + not blind to the side it built)
- [x] **NONE for R3** → open-generation is **NOT RUN** and reported as such. Mechanical
      scoring (MCQA, verifiable IFEval, answer-extracted math) needs no judge.

When a judge IS used, the win rate is reported **raw and length-controlled**
(regression at zero length difference). Instruction-only debiasing ("do not be
swayed by length") is a documented failure mode and does not count as control.

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
| 2026-07-27 | **Korean suite reduced to 2 datasets x 50 items** (`kmmlu`, `haerae_bench`; `KO_ITEMS=50`). English stays at 150/task. | Measured on the reference tier, Korean MCQA costs ~8.5 min/item on Motif — about 20x the English rate (22 s/item on gsm8k) — because the prompts are long and the reasoning is deep. Five Korean sets at 150 items would be ~100 h for Motif and ~400 h for Solar T. 50 items still clears the MIN_PAIRED_N=30 floor. Dropped: `hrm8k`, `kormedmcqa` (cost) and `click` (no `test` split — train only). Fixed before any scored run. |
| 2026-07-27 | **Sample size preregistered: 150 items per task** (lm-eval `--limit` divided per subtask: mmlu_pro=11, bbh_cot_zeroshot=6, gsm8k_cot_zeroshot=150, minerva_math=22, ifeval=150; `KO_ITEMS=150` per Korean dataset). | The full suites are ~51k KO items plus the EN tasks; at measured throughput (Motif ~22 s/item, Solar T ~89 s/item) that is hundreds of hours per model. 150 is 5x the MIN_PAIRED_N=30 floor and is fixed BEFORE any scored run, so it cannot be chosen after seeing scores. Total ≈ 1520 items per model. |
| 2026-07-26 | **Round order changed: R3 (reference) runs before R1 (floor).** | R1 is blocked: the Solar fork's server silently drops `--adapter-path` (`ModelProvider.load` keys `_adapter_map` with an already-rebound path), so F-v2 would be benchmarked **without** the rank-8 DWQ adapter it ships with — understating Solar. Held until Kimi confirms the fix or the `"adapters"`-in-body workaround. Meanwhile both reference-tier builds are ready and verified (Motif Q8 rebuilt after the mx.split defect: 5/5 greedy slices clean; Solar T public at `t384`), and a machine is idle, so R3 runs first. Round *order* is not part of the frozen decision rule — the binding commitment is that **all three rounds are disclosed regardless of outcome**, which is unchanged. |
| 2026-07-26 | **lm-eval runs zero-shot with a shared output-format instruction** (`--num_fewshot 0`, `--system_instruction`), replacing the task-default 8-shot. | Smoke showed the fewshot format measures *format imitation*, not capability, and does so asymmetrically. Solar T treated the 8 exemplars as material to analyse and answered about them — it still reached the correct number on the item inspected (`exact_match 1.0`) while the flexible extractor picked an intermediate value from its analysis (0.125 vs strict-match 0.500); Motif, which mimics the exemplar style, scored 8/8. Reporting that as a capability gap would favour the side these authors built. Zero-shot plus one instruction, identical for both, removes the confound symmetrically. Recorded before any scored run. |
| 2026-07-26 | Motif reference build replaced. | The originally shipped Motif 8-bit was corrupt (`mx.split` silent corruption above 2³¹ elements). Rebuilt, verified by generation, and re-uploaded. R3 uses the rebuilt build; its identity is recorded in the round manifest. |
