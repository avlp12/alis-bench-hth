# R3 suite + serving proposal — measured, 2026-07-27

R3 attempt 1 was aborted after 132 min. It had completed 14/154 and 12/154 items of
its *first* task. Measured cost: **~10.5 min/item/leg at `num_concurrent=4`**, which
projects to **~5.6 days per leg** for the preregistered 770 items. This document
records what was measured, what published benchmarks do instead, and what to change.

Raw numbers: `measurements/conc_{motif,solar}.json`, produced by
`harness/concurrency_probe.py`.

---

## 1. Concurrency — measured

Aggregate tokens/second is the only quantity that sets wall clock for N independent
items. Per-stream tok/s is latency, and it is *expected* to fall as batch grows.

| | c=4 | c=8 | c=16 | best |
|---|---|---|---|---|
| **Solar T** aggregate | 36.38 | 44.28 | **52.95** | c=16, **1.46×** |
| **Solar T** per-stream | 12.58 | 8.57 | 4.71 | |
| **Motif Q8** aggregate | 18.72 | 34.14 | **42.26** | c=16, **2.26×** |
| **Motif Q8** per-stream | 5.97 | 6.67 | 4.16 | |

Both models gain from more concurrency; Motif gains much more, because at c=4 it is
latency-bound rather than bandwidth-bound (53 layers, 384 experts top-8).

### This probe does NOT predict the real regime, and we can prove it

Measured at `max_tokens=2048`. The real R3 regime generates **~12,000 tokens/item**.
Two facts show the extrapolation fails:

1. Solar's per-stream rate at c=4 is **12.58 tok/s** here but was **~4.8 tok/s** in
   R3 — a 2.6× gap, from KV-cache growth over a 6× longer context.
2. Solar's aggregate at 2048 tokens is **2× Motif's** (36.4 vs 18.7), yet in R3's
   12k regime Solar was *slightly slower* (12 items vs 14 in the same 132 min).
   Whatever ranks these two at 2048 tokens does not rank them at 12k.

So the 1.46×/2.26× figures are **upper bounds**. At 12k the KV cache is 6× larger per
stream, so c=16 costs 6× more KV memory and the gain will shrink — possibly to
nothing, possibly to an OOM.

**Recommendation: `num_concurrent=8`, not 16.** It is better than 4 on both models in
the probe (Solar +22%, Motif +82%) and carries half the KV-memory risk of 16. The
scored run reveals the realized rate within its first hour; that is a cheaper
validation than another 2-hour probe at realistic length.

Even at the optimistic 1.46×/2.26×, 770 items still costs **~3.8 days** on the slower
leg. Concurrency alone does not solve this.

---

## 2. What published benchmarks do

| | items | repeats | max output tokens | temperature |
|---|---|---|---|---|
| [Artificial Analysis](https://artificialanalysis.ai/methodology/intelligence-benchmarking) v4.1 | GPQA-D **198**, CritPt **70**, AA-LCR **100**, τ³ **97**, Terminal-Bench **89**, GDPval **220** — **no subsampling** | GPQA-D **5**, Terminal-Bench 3 | **per-model**, as disclosed by each creator | 0.6 for reasoning, *unless the lab recommends otherwise* |
| [HELM Capabilities](https://github.com/stanford-crfm/helm/blob/main/docs/benchmark.md) | all scenarios downsampled to **1000** | — | — | — |
| DeepSeek / Qwen3 official | AIME **30**, GPQA-D **198** | AIME **32 samples**, GPQA-D & LCB **8**, MATH-500 **4** | 32,768 | 0.6 / top-p 0.95 |

Three conclusions that bear directly on this round:

1. **They do not slice big datasets — they pick natively small, hard ones.** Seven of
   AA's nine evals are 70–220 items and run complete. We are doing the opposite:
   mmlu_pro 12,032 → 154, bbh 6,511 → 162, minerva 5,000 → 154. That pays full
   reasoning cost per item while discarding the discriminative design the dataset was
   built around, and the resulting numbers are not comparable to any published figure.
   HELM *does* subsample — to **1000**, not 60, and not on 12k-token generations.
2. **For small n they buy power with repeats, not items.** 32 samples/question on 30
   AIME problems. [Miller, *Adding Error Bars to Evals*](https://arxiv.org/abs/2411.00640)
   says the same: resample answers to cut variance, do inference on question-level
   paired differences (we already do), and run a power analysis before trusting a
   comparison.
3. **Token and cost per task are headline numbers, not footnotes.** This validates the
   disclosed-cost ledger.

### Where our protocol diverges without justification

- **Matched `MAX_TOKENS=32768`.** AA sets this *per model, from each creator's
  disclosure*. We froze one value for both while claiming best-vs-best everywhere
  else. The value itself matches DeepSeek/Qwen convention, but nothing in our protocol
  justifies forcing both sides to the same number — and §1's probe shows this is not
  neutral: at 2048 tokens **Motif returned empty content in 3/4, 4/8 and 7/16
  requests while Solar returned 0**. Motif front-loads longer reasoning, so any shared
  cap penalises Motif specifically. A shared cap is a *matched knob that favours one
  architecture*, which is exactly what this protocol exists to avoid.
- **temp 1.0 with a single sample per item.** Both cards recommend 1.0, and AA allows
  the lab's recommendation to override — so the setting is defensible. But one sample
  at temp 1.0 is the highest-variance configuration available, and with n=1 the
  sampling noise is inseparable from item difficulty. The paired bootstrap covers
  item variance; it does not cover this.

---

## 3. Proposed suite

**Budget in tokens, not items.** An item of ifeval and an item of AIME differ by an
order of magnitude in cost; counting both as "1 item" is why the 770-item plan looked
affordable and was not.

| task | n | complete? | why |
|---|---|---|---|
| `gpqa_diamond_cot_zeroshot` | **198** | ✅ full | The standard hard-science set; AA runs it complete, so our number is directly comparable to published ones. Generative (`generate_until`, exact_match, strict-match/flexible-extract) so it works over `mlx_lm.server` — the loglikelihood variant does not. |
| `aime25` | **30** | ✅ full | Competition math, verifiable, DeepSeek/Qwen standard. 2025 rather than 2024 for contamination distance. |
| `ifeval` | **541** | ✅ full | Judge-free and rule-verifiable — the most objective scoring in the suite — and short outputs, so it is by far the cheapest discrimination per token. |
| `minerva_math500` | 500 → **150** | sampled | MATH-500 is already a curated subset of MATH, so sampling it further is the one defensible slice. Keeps a general-math axis next to AIME's hard tail. |
| `mmlu_pro` | 12,032 → **154** | sampled | Kept *only* for the broad-knowledge axis, which nothing small covers. Must be reported as a sampled estimate with its own wider CI, never as "mmlu_pro". |

**Dropped:**
- `bbh_cot_zeroshot` — 6 items per subtask across 27 subtasks is the thinnest slice in
  the suite and is meaningless per subtask; GPQA-D and AIME cover reasoning better.
- `gsm8k_cot_zeroshot` — saturated at the frontier, so it spends 150 items' budget to
  discriminate almost nothing.

Verified present in the installed lm-eval 0.4.12, all `generate_until`:
`gpqa_diamond_cot_zeroshot` n=198 · `aime24`/`aime25` n=30 · `minerva_math500` n=500
(exact_match + math_verify) · `ifeval` n=541.

### Open, and blocking the final item counts

Per-task token cost is **not measured**. The 12k figure is from mmlu_pro only; ifeval
and gsm8k are certainly far cheaper, AIME probably far more expensive. Item counts
above should be finalised against a measured tokens/item per task — a 5-item probe per
task recording `completion_tokens`, roughly one hour — not against the item counts
themselves. Setting counts before that measurement would repeat the mistake that
produced the 770-item plan.

### Also to decide

- **Per-model `max_tokens`** from each model card, replacing the matched 32768
  (AA's rule). Needs a preregistration amendment either way, including "keep it
  matched, and here is why" if that is the choice.
- **Repeats** on the small sets (AIME n=30 is far below `MIN_PAIRED_N=30` for a
  paired verdict at 1 sample). DeepSeek uses 32; even 4 would make AIME reportable and
  costs 120 generations.
