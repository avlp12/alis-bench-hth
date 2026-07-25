# RUNBOOK — running a head-to-head

Operational truth, verified by a live end-to-end dry run on 2026-07-26.
Read the ⚠ sections before the first run; each one is a failure that already happened.

---

## 0. Environment (one-time)

**lm-eval and HRET cannot share a virtualenv.** lm-eval's legacy dataset ids
(`gsm8k`) need `huggingface_hub<1.0`; HRET's `transformers` needs `hub>=1.0`.
Installing both in one env breaks whichever you touched last.

```bash
# lm-eval — isolated venv (already created as .venv-lmeval)
python3 -m venv .venv-lmeval
.venv-lmeval/bin/python -m pip install "lm-eval[api]==0.4.9" "huggingface_hub<1.0" "datasets<4"

# HRET — system env. NOTE: its vllm dep does not build on macOS; skip deps and
# install only what is actually imported.
python3 -m pip install --no-deps haerae-evaluation-toolkit
python3 -m pip install openai datasets litellm math_verify langdetect \
                       spacy scikit-learn transformers torch pandas tqdm httpx accelerate
python3 -m spacy download ko_core_news_sm
```
`run_lm_eval.sh` auto-selects `.venv-lmeval/bin/lm_eval`; override with `LMEVAL_BIN`.

## 1. Configure

```bash
cp harness/config.env.example harness/config.env && $EDITOR harness/config.env
```
Set per side: `*_MODEL` (build path), `*_REF` (8-bit anchor for KL), `*_URL`,
`*_TEMP`/`*_TOP_P`, plus `JUDGE_ENDPOINTS` (**≥2 neutral judges**, neither a contender).

⚠ **`MOTIF_MODEL` / `SOLAR_MODEL` must be the EXACT path the server was started
with.** `mlx_lm.server` *resolves* the request's `model` field: a mismatch makes it
fetch that id from HF (404) or load a second copy of the weights and OOM. This is
what killed a 312 GB server during the dry run.

⚠ **Avoid spaces in model paths** (`/Volumes/Crucial X10/...`). They survive here
but are fragile inside comma-separated `--model_args`. Symlink to a plain path.

⚠ **`MAX_TOKENS` must be generous for reasoning models.** At `max_tokens=16` a
reasoning model spends the whole budget inside its thinking block and the response
has **no `content` at all** — every item would silently score 0. `sanity_probe.py`
now fails loudly on this ("empty content with N chars of reasoning").

## 2. Serve (one model per box)

```bash
harness/serve.sh motif      # Gesicht :8081  (uses the @motif fork via PYTHONPATH)
harness/serve.sh solar      # Epsilon :8082  (set SOLAR_FORK if it needs one)
```
First request triggers the weight load: **~8 minutes for a 312 GB build off an
external SSD.** Don't mistake that for a hang. Memory is shared with anything else
running (a concurrent quantization job will OOM a large server — schedule around it).

## 3. Preflight — refuses a bad setup

```bash
harness/preflight.sh
```
Checks harnesses, builds, endpoints (template + KO-consistency), ≥2 judges,
**runtime provenance (clean git tree)**, and disk. Fix every ✗ before continuing.

## 4. Run

```bash
harness/run_all.sh                        # sanity → lm-eval → HRET → judge → integrity gate
harness/run_kl.sh motif ; harness/run_kl.sh solar     # where each build + its ref live
# throughput: ONE box, back-to-back, warm-up is automatic
harness/serve.sh motif & sleep 60; python3 harness/throughput_probe.py --name motif --url http://localhost:8081; kill %1
harness/serve.sh solar & sleep 60; python3 harness/throughput_probe.py --name solar --url http://localhost:8082; kill %1

python3 harness/manifest.py check && python3 harness/aggregate.py
```

`run_all.sh` is **fail-closed**: every step's exit code is recorded, every produced
cell is stamped with the run id, and aggregation is blocked unless all required
cells are present and fresh. `ALLOW_PARTIAL=1` produces a report clearly marked
PARTIAL — that is a debugging artifact, not a result.

Set `LIMIT=20` for a fast rehearsal of the whole chain.

## 5. Reading the report

- **Per-task verdict** = paired bootstrap CI clearing `EQUIV_MARGIN` **and** Holm
  significance across the task family. Anything else is `INCONCLUSIVE`. There is no
  single headline number by design.
- **Disclosed cost** — settings, runtime commit, and tokens actually spent. A win
  bought with 3× the tokens is reported as exactly that.
- **KL** is per-model (`own 8-bit ‖ build`). **Never rank it across models.**
- `NO PAIRED SAMPLES` means the statistical layer is off for that task — marginal
  means alone cannot decide a winner. Re-run with `--log_samples`.

## 6. Known gaps (state them, don't paper over them)

| gap | status |
|---|---|
| KMMLU-Redux / KMMLU-Pro / KoSimpleQA | **not in HRET 0.1.0** — plain `kmmlu` is used and auto-flagged `_contaminated`; treat as a lower bound |
| GPQA-Diamond, LiveCodeBench, HumanEval+/MBPP+, AIME/HMMT, RULER | not automated here — run from their own repos, do not claim them |
| self-consistency / best-of-k | not implemented; single sample per item both sides |
| Korean-native judge calibration | required before open-gen results are publishable |

## 7. Dry-run reference (2026-07-26)

Two `mlx_lm.server` instances of a 279 MB `Qwen2.5-0.5B-Instruct-4bit`, one per
"side", exercised the whole chain. Same model both sides ⇒ `gsm8k_cot`
Δ=+0.000, p=1.000 → **INCONCLUSIVE** (a bias bug would have invented a winner);
filter selection picked `strict-match` per schema; judge run counted
parse_fail 3 / order_effect 1 / tie 1 separately and returned
`SMOKE (not publishable)` rather than a result.
