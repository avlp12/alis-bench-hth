#!/usr/bin/env bash
# run_lm_eval.sh <motif|solar>
# English / reasoning / math / instruction-following over the OpenAI-compatible
# endpoint, using ONLY generative (generate_until) tasks — these score by answer
# extraction or verifiable rules, so they work over mlx_lm.server.
#
# Why not loglikelihood MCQA here: mlx_lm.server does not reliably return
# echo+logprobs, so kmmlu/haerae/kobest logprob scoring would silently break.
# Korean MCQA is handled by run_hret.py (generative + parsing + language penalty).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/config.env"
NAME="${1:?motif|solar}"
case "$NAME" in
  motif) URL="$MOTIF_URL"; MODEL="$MOTIF_MODEL"; TEMP="$MOTIF_TEMP"; TOP_P="$MOTIF_TOP_P";;
  solar) URL="$SOLAR_URL"; MODEL="$SOLAR_MODEL"; TEMP="$SOLAR_TEMP"; TOP_P="$SOLAR_TOP_P";;
  *) echo "usage: run_lm_eval.sh motif|solar"; exit 1;;
esac
OUT="$RESULTS_DIR/lm_eval/$NAME"; mkdir -p "$OUT"

# best-vs-best: generous CoT budget for peak/thinking; probes log tokens actually spent
GEN_TOKS="$MAX_TOKENS"

# Generative, contamination-aware core set (all confirmed present in lm-eval 0.4.9):
#   mmlu_pro   -> hard knowledge (answer-extraction)
#   bbh        -> Big-Bench-Hard CoT
#   gsm8k_cot  -> grade-school math CoT
#   minerva_math -> MATH (competition math)
#   ifeval     -> verifiable instruction following (judge-free -> most objective)
# Purpose-built ZERO-SHOT variants — their filters expect chat-style output.
# (bbh/gsm8k_cot are the *fewshot* configs; forcing them to 0-shot leaves filters
#  and stop-sequences tuned for exemplar mimicry, which scores format-fit.)
# ifeval is EXCLUDED here and run separately: it scores per-item instruction
# following, so any injected global format instruction corrupts it by construction.
TASKS="${TASKS:-mmlu_pro,bbh_cot_zeroshot,gsm8k_cot_zeroshot,minerva_math}"

echo "[lm_eval] $NAME tasks=$TASKS temp=$TEMP top_p=$TOP_P max_gen_toks=$GEN_TOKS (peak) -> $OUT"
# --apply_chat_template is REQUIRED by local-chat-completions (it asserts the
# request is a message list, not a string). Combined with tokenized_requests=False
# the messages are sent as JSON and THE SERVER applies each model's OWN chat
# template — exactly the fairness property this protocol requires.
# seed must go in --model_args (that's what reaches the API payload); the CLI
# --seed only seeds python/numpy/fewshot sampling.
# lm-eval and HRET have MUTUALLY EXCLUSIVE pins (lm-eval's legacy dataset ids need
# huggingface_hub<1.0; HRET's transformers needs hub>=1.0) — each runs from its own
# venv. Verified live 2026-07-26.
LMEVAL_BIN="${LMEVAL_BIN:-$here/../.venv-lmeval/bin/lm-eval}"
[ -x "$LMEVAL_BIN" ] || { echo "[lm_eval] venv missing, falling back to PATH lm-eval"; LMEVAL_BIN="lm-eval"; }
# lm-eval >=0.4.10 moved evaluation under a `run` subcommand (breaking change).
LMEVAL_RUN=(run)
"$LMEVAL_BIN" run --help >/dev/null 2>&1 || LMEVAL_RUN=()   # pre-0.4.10 fallback
# ZERO-SHOT + an explicit output-format instruction, applied IDENTICALLY to both
# contenders. Rationale (preregistration amendment 2026-07-26): the 8-shot format
# measures how well a model imitates the fewshot phrasing, not whether it solves
# the problem. Verified in smoke: Solar T read the 8 exemplars as material to
# analyse and answered *about* them — it reached the right number (exact_match 1.0
# on the item inspected) while the flexible extractor picked an intermediate number
# from its analysis (0.125 vs strict 0.500), whereas Motif, which mimics the format,
# scored 8/8. Publishing that gap would report format-fit as capability, in the
# direction that favours the side we built. Zero-shot + one shared format
# instruction removes the confound symmetrically.
FEWSHOT="${FEWSHOT:-0}"
# The instruction must match what the extractors ACTUALLY parse, verified against
# the installed task configs (2026-07-26):
#   bbh_cot_zeroshot / mmlu_pro : "(?<=the answer is )" / "answer is \(?([A-J])\)?"
#   gsm8k_cot_zeroshot strict   : "The answer is (\-?[0-9\.\,]+)."
# A "#### <answer>" instruction (an earlier attempt) parses on NONE of them and
# would have scored obedience to our own instruction as failure.
FORMAT_INSTRUCTION="${FORMAT_INSTRUCTION:-End your reply with a final line of the form: The answer is <answer>.}"

"$LMEVAL_BIN" "${LMEVAL_RUN[@]}" --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${URL}/v1/chat/completions,num_concurrent=${NUM_CONCURRENT:-4},max_retries=3,tokenized_requests=False,timeout=7200,seed=${SEED}" \
  --tasks "$TASKS" ${LIMIT:+--limit "$LIMIT"} \
  --num_fewshot "$FEWSHOT" \
  --system_instruction "$FORMAT_INSTRUCTION" \
  --apply_chat_template \
  --gen_kwargs "temperature=${TEMP},top_p=${TOP_P},max_gen_toks=${GEN_TOKS}" \
  --seed "$SEED" --batch_size 1 --log_samples \
  --output_path "$OUT"

# ifeval — SEPARATE run, NO --system_instruction. It scores each item against its
# own constraints (end-with-phrase, wrap-in-quotes, JSON-only, language-only...),
# so a global format instruction makes it measure system-vs-user priority instead
# of instruction following, penalising the more system-obedient model.
echo "[lm_eval] $NAME ifeval (no system instruction, native zero-shot)"
"$LMEVAL_BIN" "${LMEVAL_RUN[@]}" --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${URL}/v1/chat/completions,num_concurrent=${NUM_CONCURRENT:-4},max_retries=3,tokenized_requests=False,timeout=7200,seed=${SEED}" \
  --tasks ifeval ${LIMIT:+--limit "$LIMIT"} \
  --apply_chat_template \
  --gen_kwargs "temperature=${TEMP},top_p=${TOP_P},max_gen_toks=${GEN_TOKS}" \
  --seed "$SEED" --batch_size 1 --log_samples \
  --output_path "$OUT"

python3 - "$OUT" "$TASKS" <<'PY'
import json, sys
out, tasks = sys.argv[1], sys.argv[2].split(",")
json.dump({"tasks_run": tasks,
  "advertised_but_NOT_automated_here": [
    "gpqa_diamond (loglikelihood — run on the bf16/8-bit ref via HF backend)",
    "LiveCodeBench (external repo; point at the same base_url)",
    "HumanEval+/MBPP+ (code exec — separate, --confirm_run_unsafe_code)",
    "AIME/HMMT (add explicitly if in scope)",
    "RULER / long-context (separate harness)"]},
  open(out + "/_coverage.json", "w"), ensure_ascii=False, indent=2)
print("[lm_eval] wrote _coverage.json — 'advertised != automated' is now explicit")
PY
echo "[lm_eval] $NAME done."
# ---------------------------------------------------------------------------
# HumanEval / MBPP (code, executes generated code -> explicit opt-in required):
#   lm_eval --model local-chat-completions \
#     --model_args "model=${MODEL},base_url=${URL}/v1/chat/completions,num_concurrent=4,tokenized_requests=False" \
#     --tasks humaneval,mbpp --confirm_run_unsafe_code \
#     --gen_kwargs "temperature=0.2,max_gen_toks=1024" --seed $SEED --output_path "$OUT/code"
#
# LiveCodeBench (contamination-resistant, RECOMMENDED) is not in lm-eval 0.4.9 —
# run it from its own repo pointed at the same base_url. See README.
# GPQA-Diamond is loglikelihood in 0.4.9; run it on the bf16/8-bit ref via the HF
# backend, or skip for the endpoint run (documented in README).
