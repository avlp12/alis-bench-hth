# HANDOVER — read this first after a context reset

Written 2026-07-26. Everything below is verified state, not plan. Where something is
believed-but-unverified it says so.

---

## 1. Where we are in one paragraph

We are running a 3-round head-to-head between **Motif-3-Beta** (ours) and **Solar
Open 2** (Kimi's), each quantized by its own team, served locally over MLX, published
round-by-round. **Nothing has been scored yet.** The harness is at
`~/alis-bench-hth` (public: https://github.com/avlp12/alis-bench-hth). R3 (reference
tier) is configured, both servers are up, the stage gate passes, and the
preregistration is filled and hash-frozen. **Four adversarial reviews have run; all
four returned BLOCK and all four were right.** Round 4's findings are fixed and the
stage gate passes with two new checks that reproduce them (§6). R3 may start.

## 2. Live state right now

| what | where | status |
|---|---|---|
| Motif-3-Beta 8-bit (rebuilt) | epsilon `/Users/m3ms/motif3/models/Motif-3-Q8`, served `10.0.0.2:8081` | ✅ loaded, verified |
| Solar Open 2 "T" (Q8) | gesicht `~/Documents/kimi/workspace/builds/T-q8`, served `127.0.0.1:8082` | ✅ loaded |
| R3 config | `cases/2026-07-motif3-vs-solar2/round3-reference.env` (copied to `harness/config.env`) | ✅ frozen |
| Preregistration | `cases/2026-07-motif3-vs-solar2/PREREGISTRATION.md` + `.sha256` = `6f7c7af7ebee34d5…` | ✅ frozen |
| Stage gate | `harness/stage_gate.py` | ✅ PASS (5 layers) |

**epsilon is reachable only as `10.0.0.2` over the Thunderbolt bridge.** Its hostname
is misspelled `episilon` in DNS; `epsilon.local` does not resolve. SSH: `ssh 10.0.0.2`.

Servers are large (334 GB + 266 GB). They cannot co-reside on one box — one model per
machine. Do not start a second big model on either host without stopping the first.

## 3. What must happen next, in order

1. `harness/preflight.sh` → `harness/run_all.sh` → `manifest.py check` → `aggregate.py`.
   Adversarial review is **done** (4 rounds, §6). Stage gate passes including the new
   stage 6.
2. Publish R3 **whatever it says** (binding rule in the prereg), then R2 (mid), then R1
   (floor) once Kimi's adapter bug is resolved.
3. **Before R1:** implement deterministic Korean subsampling. `KO_ITEMS` cannot be
   handed to HRET 0.1.0 (no sampling parameter anywhere in its API), so `run_hret.py`
   now *aborts* rather than silently scoring all ~35k kmmlu items. R3 is unaffected
   (`KO_SUITE=off`).

## 3b. Sample size — PREREGISTERED at 150 items per task

The full suites are ~51k Korean items plus the English tasks; at measured throughput
(Motif ~22 s/item, **Solar T ~89 s/item**) running everything is hundreds of hours per
model. The preregistered sample is **150 items per task**, frozen before any scored run
— 5× the `MIN_PAIRED_N=30` floor, so verdicts stay possible.

lm-eval's `--limit` applies **per subtask**, so group tasks need a divided limit:

| task | subtasks | `--limit` | total |
|---|---|---|---|
| `mmlu_pro` | 14 | 11 | ~154 |
| `bbh_cot_zeroshot` | 27 | 6 | ~162 |
| `gsm8k_cot_zeroshot` | 1 | 150 | ~150 |
| `minerva_math` | 7 | 22 | ~154 |
| `ifeval` | 1 | 150 | ~150 |

English total ≈ 770 items at ~22 s/item.

**R3 is ENGLISH ONLY (`KO_SUITE=off`); Korean is measured in the floor round.**
⚠ **litellm's default timeout is 600 s — SHORTER than one Korean generation here.**
Every item timed out and retried 3x (~90 min/item) and the error text was returned as
the prediction. Set `HRET_TIMEOUT=3600`. The error guard caught it and aborted rather
than scoring timeout strings as answers — that guard is load-bearing, do not remove it.

**Korean is a different cost regime — measure, do not extrapolate from English.**
Korean MCQA on the reference tier runs **~8.5 min/item** on Motif (≈20× the English
rate: long prompts, deep reasoning). So the Korean suite is **2 datasets × 50 items**
(`kmmlu`, `haerae_bench`; `KO_ITEMS=50`) = 100 items ≈ 14 h on Motif. Dropped `hrm8k`
and `kormedmcqa` for cost, and `click` because it has **no `test` split** (train only)
and would have failed mid-run.

## 4. Hard-won gotchas — do not rediscover these

**Serving**
- `mlx_lm.server` RESOLVES the request's `model` field. If it does not exactly equal
  the served path it tries to fetch that id from HF (404) or loads a second copy and
  OOMs. This killed a 312 GB server.
- A reasoning model can spend the whole budget thinking and return **no `content` at
  all**. Scoring `content` alone then yields 0 on every item. `sanity_probe.py` fails
  loudly on this now.
- Both forks split reasoning into `reasoning_content`, so extraction is symmetric.
  Motif reasons internally (~371 chars) and answers concisely; Solar T reasons in the
  visible answer. That is model behaviour, disclosed — not a condition asymmetry.
- **Kimi's fork drops `--adapter-path` silently.** `ModelProvider.load` rebinds
  `model_path` from `"default_model"` to the real path and *then* keys `_adapter_map`
  with the rebound path → always misses → no LoRA. Workaround: put
  `"adapters": "<path>"` in the request body. This blocks R1 (F-v2 ships a rank-8 DWQ
  adapter); reported to Kimi in `cases/.../to-kimi-2026-07-26-adapter-bug.md`.

**Harnesses**
- lm-eval and HRET have **mutually exclusive pins** (lm-eval's legacy dataset ids need
  `huggingface_hub<1.0`; HRET's transformers needs `>=1.0`). lm-eval lives in
  `.venv-lmeval`; HRET in the system env. Never merge them.
- lm-eval ≥0.4.10 moved evaluation under a `run` subcommand and 0.4.12 changed
  `TaskManager.load()` to a flat dict. We are on **0.4.12**; rollback backup is
  `.venv-lmeval.bak-0.4.9`.
- `--apply_chat_template` is **required** by `local-chat-completions` (it asserts).
  With `tokenized_requests=False` the server applies each model's own template — which
  is exactly the fairness property we want.
- API payload `seed` comes from `--model_args seed=`, not the CLI `--seed`.
- HRET dataset **registry keys** ≠ module names. Real keys: `kmmlu`, `haerae_bench`,
  `hrm8k`, `click`, `kormedmcqa`, `KUDGE`. (`haerae`/`kormedqa` do not exist.)
  `kmmlu_redux`/`kmmlu_pro`/`kosimpleqa` are NOT in HRET 0.1.0 — plain `kmmlu` is used
  and auto-flagged `_contaminated`.
- HRET's `LiteLLMBackend` needs `provider="openai"` **and** `model_name="openai/<path>"`
  (the prefix must be inside model_name; it only auto-prefixes azure/bedrock). Without
  it, litellm fails and the backend returns the error **as the prediction string** — a
  silent ~0% for the whole Korean suite. There is now a guard that aborts on this.

**Scoring**
- Zero-shot + a shared format instruction, NOT few-shot: the fewshot format measures
  imitation, and it broke asymmetrically (Solar analysed the exemplars instead of
  answering). This matches OpenAI simple-evals' published rationale.
- The instruction must match what the extractors actually parse:
  `The answer is <answer>.` A `#### <answer>` instruction parses on **none** of the
  tasks and would score obedience to our own instruction as failure.
- **ifeval runs separately with NO system instruction** — it scores per-item
  constraints, so a global format instruction corrupts it by construction.
- Use the purpose-built zero-shot task variants: `bbh_cot_zeroshot`,
  `gsm8k_cot_zeroshot`. `minerva_math` primary metric is **`math_verify,none`** (its
  `exact_match` needs the Minerva incantation only its 4-shot exemplars taught).
- `num_concurrent` above 4 does **not** help — MoE decode is memory-bandwidth bound and
  different sequences route to different experts. 8 was slightly slower than 4.
- Throughput: Motif ~22 s/item, **Solar T ~89 s/item** (4×). Budget accordingly and
  report it as disclosed cost.

## 5. The 8-bit incident (closed, but its lessons drive the protocol)

Our published 8-bit build was corrupt: the fused expert `gate_up_proj` is 4.03e9
elements = 1.88× the 2³¹ limit where `mx.split` silently corrupts past the 4 GiB
offset (ml-explore/mlx#3836, **now CLOSED upstream 2026-07-12**), corruption starting
at expert 205/384. The build predated our own strided-slice workaround and was never
re-made — because a rule *we wrote* ("forward-math fixes need no re-quantization") was
misapplied to what was actually a **weight-materialization** fix.

Rebuilt, verified (5/5 greedy slices; old and new byte-identical below the corruption
offset), re-uploaded, card updated, [discussion #1] answered and **closed**. Knock-on:
the `KL vs Q8` figures on the 2.3bpw/4.5bpw cards were measured against the corrupt
reference and are **withheld pending re-measurement** (models and port parity are
unaffected). Postmortem written into the alis-dwq case study.

**Open follow-up:** re-measure those KL numbers now that a trustworthy anchor exists.

## 6. Adversarial review — 4 rounds, all BLOCK, all fixed

Rounds 1–3 are in git history. **Round 4 (final gate) found two CRITICAL defects of the
exact same shape as round 3's: a fix written into a file that no code path reads.**

1. **Per-task limits were inoperative.** `LIMIT_MMLU_PRO`/`LIMIT_BBH`/… were exported by
   `config.env` and read by *nothing*; `run_lm_eval.sh` passed a single `--limit
   "$LIMIT"` that was unset. The "150 items/task" plan would have run **~25k items per
   model (~3 weeks)**. One invocation with `--tasks a,b,c` cannot express per-task
   limits at all — each task now runs separately, and a task with no limit **aborts**
   unless `LIMIT_FULL_OK=1`.
2. **bbh was silently dropped from the verdict family.** `METRIC_SCHEMA` wanted
   `exact_match,get-answer`, which exists only in the cot_**fewshot** config; the
   zero-shot group aggregates over `flexible-extract`. 4 tasks would publish where 5
   were frozen — and the round would still say COMPLETE.
3. A COMPLETE report **never said the Korean leg was NOT RUN** (the manifest recorded
   it; `aggregate.py` never rendered it). Absence read as parity. Now an explicit
   section.
4. A leftover `kmmlu_items.json` was ingested with no run_id/freshness check and
   produced a Korean row inside the English-only round. `KO_SUITE=off` now makes
   `aggregate.py` ignore the `hret/` tree entirely and log what it skipped.
5. `judge_pairwise.bootstrap_sign` emitted a direction at any n while `aggregate` refused
   below 30. Same floor now applies to both.
6. `preflight.sh` hard-failed on 0 judges — which is R3's *preregistered* choice, so the
   mandated gate always failed and trained the operator to bypass it. 0 judges warns;
   **1** judge still fails.

**Two new stage-gate checks reproduce #1 and #2 as failures** (`stage_gate.py` stage 6):
every `METRIC_SCHEMA` key is validated against the *installed* task configs, and the
sampling plan is resolved through `run_lm_eval.sh`'s own code path (`PLAN_ONLY=1`) and
must total ~770 items/model. Neither defect was visible by reading the harness — only by
asking the installed package and the actual script what they resolve to. **Add a gate
check for every fix that lives in a config file.**

Round 4 also *confirmed working*, by execution: the Korean per-item extraction (40/40,
keys identical across runs), the shared format instruction against all five lm-eval
0.4.12 extractors, and `HRET_TIMEOUT=3600`/`MAX_TOKENS=32768` reaching the litellm call.

### P0 fixes from earlier rounds (all still in force)

1. `EQUIV_MARGIN` 0.0 → **0.015** — framework noise alone is 1–2pp.
2. `MAX_TOKENS` 8192 → **32768** — the reasoning-model standard; counting truncation
   is not preventing it.
3. Preregistration filled + **hash-frozen**.
4. **Length-controlled win rate** in `judge_pairwise.py` (AlpacaEval-2.0 style: logistic
   regression on normalized length difference, evaluated at zero). Reported alongside
   the raw rate, never instead of it. Instruction-only debiasing is a documented
   failure mode.
5. **Korean suite paired** — `run_hret.py` persists per-item scores; `aggregate.py`
   runs them through the same bootstrap + Holm family as English.

## 7. Prior-art gaps still open (P1/P2, from the literature review)

Not blocking, but decide before publishing:
- **MDE / power** not reported — print per-task minimum detectable effect next to every
  INCONCLUSIVE so "inconclusive" ≠ "underpowered".
- **Single sample at temp 1.0** — below the multi-seed standard for small sets;
  currently disclosed as a limitation.
- **KL reported as a bare mean** — llama.cpp convention also reports same-top-1 % and
  percentile KLD; "Accuracy is Not All You Need" adds % flips.
- **Format-sensitivity unmeasured** — a 2–3 variant robustness appendix on ~100 items.
- **Contamination table covers only kmmlu** — add per-task status (GSM8K/GSM1k, BBH
  saturation, MMLU-Pro static, IFEval constraint overfitting, KO suites unaudited) and
  state that symmetric exposure is the honest defence.
- **Whole-run token ledger** — `failure_rates()` already streams every samples file;
  accumulate output tokens/chars per task per model.
- **Weight checksums in the manifest** — a filesystem path is not externally
  verifiable (Reflection-70B is the cautionary tale).

## 8. Tooling rules

- **codex must be called with** `--ignore-user-config --ignore-rules --ephemeral
  --skip-git-repo-check -s read-only --output-schema <file>`. Without these it inherits
  `~/.codex/config.toml`, project `.rules` and prior session summaries and answers a
  different question — it did this three times. Output schemas need
  `additionalProperties: false` on **every** object. `-s read-only` has **no network**,
  so codex cannot do web research; use a web-capable agent for literature work.
- Toolchain is checked weekly (`TOOLCHAIN.md` has the version table, backups and
  restore commands). Instruments are upgraded **only before a scored run**.

## 9. Key files

```
~/alis-bench-hth/
  HANDOVER.md            <- this file
  TOOLCHAIN.md           versions, backups, restore, codex invocation rules
  harness/stage_gate.py  verify every layer in one pass — run this first
  harness/preflight.sh   refuses a bad setup
  harness/run_all.sh     fail-closed orchestrator
  harness/manifest.py    run-id integrity, disclosure ledger
  harness/aggregate.py   metric schema, paired bootstrap + Holm, plumbing failures
  harness/judge_pairwise.py  ≥2 judges, both orders, length control
  harness/blind_pack.py  judge-free blinded pack for a human reviewer
  cases/2026-07-motif3-vs-solar2/
    PREREGISTRATION.md(.sha256)  frozen
    round3-reference.env / round1-floor.env
    to-kimi-2026-07-26-adapter-bug.md
~/motif3/                Motif builds + scripts (convert_motif.py, motif_quant_predicate.py)
~/alis-dwq/              quantization pipeline + the motif-3-beta case study/postmortem
~/glm5.2/mlx-lm          our fork @2a07e51 (clean) — Motif model class
~/Documents/kimi/workspace/mlx-lm   Kimi's fork @7937fda — Solar model class
```

## 10. Standing instructions from the user

- best-vs-best: each model at its **own** optimal quantization and peak settings; never
  force matched recipes. Objectivity comes from invariant task/scoring/judge + full
  disclosure of what each side spent.
- Each round is **published as it completes**, win or lose.
- Adversarial verification before running; fix, then verify again (≥2 more rounds).
- Public comments are first person singular ("I", not "we").
- Do one thing at a time; do not saturate both machines.
