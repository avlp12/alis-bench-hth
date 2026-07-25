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
TASKS="mmlu_pro,bbh,gsm8k_cot,minerva_math,ifeval"

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
LMEVAL_BIN="${LMEVAL_BIN:-$here/../.venv-lmeval/bin/lm_eval}"
[ -x "$LMEVAL_BIN" ] || { echo "[lm_eval] venv missing, falling back to PATH lm_eval"; LMEVAL_BIN="lm_eval"; }
"$LMEVAL_BIN" --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${URL}/v1/chat/completions,num_concurrent=4,max_retries=3,tokenized_requests=False,timeout=7200,seed=${SEED}" \
  --tasks "$TASKS" ${LIMIT:+--limit "$LIMIT"} \
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
