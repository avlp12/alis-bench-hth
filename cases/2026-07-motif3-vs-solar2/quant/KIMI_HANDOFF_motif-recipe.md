# Handoff to Kimi — Motif-3-Beta quantization recipe + the head-to-head pipeline

> **사용자용 요약:** Solar는 Kimi가 **자기 최적 방식**으로 양자화한다. 아래는 (1) 우리가 Motif-3-Beta를
> 어떻게 양자화했는지 — **참고·영감용 레시피**, (2) 두 모델을 공정하게 붙이기 위한 **비교가능성 계약**.
> 레시피를 베끼라는 게 아니라, Solar에 맞게 **능가하되 산출물을 비교가능하게** 만들라는 것.

---

## 0. Ground rules — this is a *best-vs-best* head-to-head

- Each model competes at its **peak**. Quantize Solar however is optimal **for Solar** — **beat** this recipe, don't copy it.
- Objectivity comes from **invariant task / scoring / judge + full disclosure**, not from matched knobs.
- The only hard requirements are the **comparability contract (§3)**. Everything else is yours to optimize.

---

## 1. What we did to Motif-3-Beta (reference recipe, via [alis-dwq](https://github.com/avlp12/alis-dwq))

**Arch recap:** 314B / ~13B-active MoE · 384 routed + 1 shared, top-8 · GDLA attention (MLA-style low-rank q/kv
+ differential-v2 + elementwise gate) · **PolyNorm** activation · **mHC** 4-wide hyper-connections · interleaved
SWA (window 129) · 262K YaRN · text-only.

**Pipeline:** census → static builds (predicate) → clip-search requant → **layerwise DWQ** → KL eval.

**Bit profiles** (affine; experts dominate size at 97.84% of params):

| profile | experts | attn / shared / dense | embed+head | chokepoint | size |
|---|---|---|---|---|---|
| FLOOR | 2-bit / g128 | 4-bit | 6-bit | 8-bit | ~91 GB · 2.31 bpw |
| C6 (golden / DWQ teacher) | 4-bit / g64 | 6-bit | 6-bit | 8-bit | ~179 GB · 4.55 bpw |
| Q8 (reference) | 8-bit / g64 | 8-bit | 8-bit | 8-bit | ~334 GB |

**Predicate logic** (per-tensor bit assignment): routed experts → low bits · **router weight + expert-bias → kept
fp** (discrete top-8) · latent chokepoint (`wkv_a`, feeds all KV heads) → 8-bit · embed/head → 6-bit · tiny
residual-mixing / differential-λ projections → kept fp · norms/PolyNorm params → no `to_quantized` anyway.

**Calibration:** self-generated, coverage-stopped via routing hooks. Mix **EN 0.30 / code 0.25 / KO 0.45**,
chat-templated (calibrate in the distribution you serve).

**DWQ:** two-phase layerwise (dump teacher targets → train student), **K=8 layers/round with round rollback**,
teacher = C6, **`lr 1e-5`** (the ~85 GB floor student tolerates it; ~`1e-6` for larger ~240 GB students —
scale lr to student size), `seed 7`, `max-seq 512`. Path: raw quant → **clip-search** (fixes greedy degeneration)
→ **DWQ** (cuts KL **58–72%** vs the 8-bit reference).

---

## 2. Hard-won lessons — these will bite Solar too

- **"Config present but unread."** Motif's incoherence was **PolyNorm** needing `sigmoid(weight) × 0.5` + bias
  clamp — flags that existed in config but the modeling code never read. For Solar: **audit every activation /
  norm / scaling flag in config and confirm your MLX port actually reads it.** Diagnostic: garbage from a single
  BOS token (position 0, softmax=1) ⇒ **position-independent** defect ⇒ suspect activations, **not** rope.
- **`mx.split` silently corrupts >2³¹-element tensors** ([ml-explore/mlx#3836](https://github.com/ml-explore/mlx/issues/3836)) — use strided slices for any big fused `gate_up`.
- **Router / expert-bias must stay fp** — quantizing them breaks discrete top-8 routing.
- **Linear-attention state projections** (Solar-specific) are your **chokepoint** analog of Motif's `wkv_a` —
  the recurrent state carries the whole sequence; give it extra bits and keep its decay/gate params fp.
- **Verify parity first.** KL ≈ 1e-7 vs a torch reference on a truncated few-layer forward **before** trusting
  any quant number. (Solar's shipped modeling code may have its own bugs — check it like we checked Motif's.)

---

## 3. Comparability contract — the ONLY hard requirements

For the head-to-head to be fair while each model stays at its own optimum, each Solar build must:

1. **Serve via `mlx_lm.server`** (OpenAI-compatible `/v1/chat/completions`) — same serving engine as Motif.
2. **Ship its own correct chat template** — each model uses its own; that's fair, not a confound.
3. **Provide an 8-bit (or bf16) reference build** → so we can compute **per-model `KL(own-ref ‖ build)`**, the
   *disclosed quantization cost* of your build. (Reported **alongside** results, never used to normalize across models.)
4. **Be the best build runnable on the target box.** Want the tier curve? Ship a couple of tiers (e.g. a golden +
   a floor), each independently optimized — not forced to Motif's bpw.
5. **Disclose** Solar's recipe + per-tier **bpw / GB** + per-tier **KL**. Peak-vs-peak is only honest if the cost
   each model paid (memory, and in §4 also thinking tokens) is on the table.

That's it. Match **nothing** else — optimize freely.

---

## 4. The eval Solar will face (optimize toward it honestly)

Head-to-head kit lives in `motif3/h2h/` (see its `README.md`). Each model runs at **peak**: thinking on, its own
recommended sampling, its own best prompt — all **disclosed**.

- **Korean** (via HRET): KMMLU-Redux/Pro, HAE-RAE, HRM8K, CLIcK, KoSimpleQA — with a Korean **language-consistency
  penalty** (answering a KO prompt in English is penalized — watch Solar's code-switching under low bits).
- **EN reasoning/math/IF** (lm-eval generative): MMLU-Pro, BBH, GSM8K, MATH, IFEval.
- **KO open-gen** (pairwise, neutral position-debiased judge): LogicKor / Arena-Hard.
- **Efficiency** (same box): decode tok/s **and chars/s** — Solar's KO-efficient tokenizer helps here, so we
  compare **chars/s**, not tok/s.
- **Quant cost**: `KL(own 8-bit ‖ build)` per model (§3.3).

---

## 5. Reuse our tooling? Optional.

`alis-dwq` is open and modular — `clip_quantize`, `layerwise` (DWQ monkey-patch over stock mlx-lm ≥0.31),
`gen_calib` (coverage-stopped calibration), `eval_kld`, `expert_traffic`. Reuse as-is, adapt, or ignore. **Not
required** — your own quantizer is entirely fine as long as §3 holds. If you do reuse it, the only Solar-specific
work is a predicate keyed to Solar's tensor names (experts / router / linear-attn chokepoint) and a `convert_solar.py`.
