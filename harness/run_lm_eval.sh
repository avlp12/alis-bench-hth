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

# Suite (amended 2026-07-27, after the token-cost probe): natively small, hard,
# generative sets run COMPLETE where affordable; big sets get a SEEDED-RANDOM
# frozen sample, never a head-of-dataset --limit slice. Published-benchmark
# practice (Artificial Analysis, HELM) runs small sets whole and never slices
# thin; that is what makes our gpqa/aime numbers directly comparable.
#   gpqa_diamond_cot_zeroshot -> hard science, run COMPLETE (198), AA-comparable
#   aime25                    -> competition math, COMPLETE (30), boxed-answer scoring
#   mmlu_pro                  -> broad knowledge, seeded sample (nothing small covers it)
#   minerva_math500           -> general math, seeded sample of the curated 500
#   ifeval                    -> rule-verifiable instruction following, seeded sample
TASKS="${TASKS:-gpqa_diamond_cot_zeroshot,aime25,mmlu_pro,minerva_math500,ifeval}"

# ---- per-task item plan ---------------------------------------------------
# plan values: "full" (whole dataset, on purpose), "samples" (frozen seeded
# indices from $SAMPLES_FILE via --samples), or an integer (--limit; smoke only —
# --limit takes the FIRST N docs, which is not a sample).
# A bare LIMIT overrides everything: that is the smoke/stage-gate path.
# Fail closed: a task with no plan aborts — round 4 proved a plan that lives
# only in a config file nothing reads is how ~25k items nearly got scored.
plan_for() {
  [ -n "${LIMIT:-}" ] && { echo "$LIMIT"; return; }
  local v
  case "$1" in
    gpqa_diamond_cot_zeroshot) v="${PLAN_GPQA:-}";;
    aime25)                    v="${PLAN_AIME25:-}";;
    mmlu_pro)                  v="${PLAN_MMLU_PRO:-}";;
    minerva_math500)           v="${PLAN_MINERVA:-}";;
    ifeval)                    v="${PLAN_IFEVAL:-}";;
  esac
  if [ -z "$v" ]; then
    echo "[lm_eval] FATAL: no item plan for task '$1' — set PLAN_<TASK> in config.env" >&2
    echo "ABORT"; return
  fi
  echo "$v"
}

# The frozen sample: {leaf_task: [doc indices]} drawn by draw_samples.py, hash
# in the preregistration. --samples keys on LEAF names, so extract this task's
# leaves from the frozen file (passing other tasks' entries is undefined).
samples_json_for() {
  python3 - "$SAMPLES_FILE" "$1" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); t = sys.argv[2]
sub = {k: v for k, v in m.items() if k == t or k.startswith(t + "_")}
sys.exit(f"no entries for {t} in samples file") if not sub else print(json.dumps(sub, separators=(",", ":")))
PY
}

# Fairness is "same instruction for both MODELS", not "one instruction for all
# TASKS": each task gets the instruction its INSTALLED extractor actually parses
# (round-4 method: ask the installed config, don't assume).
#   aime25's process_results extracts ONLY $...$ / \boxed{} — "The answer is X."
#   parses on none of it, so the shared line would score obedience as failure.
#   ifeval scores per-item instructions; ANY global instruction corrupts it.
instruction_for() {
  case "$1" in
    aime25) printf '%s' 'End your reply with your final answer in \boxed{}.';;
    ifeval) printf '';;
    *)      printf '%s' "$FORMAT_INSTRUCTION";;
  esac
}

# PLAN_ONLY=1 resolves and prints the sampling plan through THIS code path, then
# exits without generating. stage_gate.py asserts on it — the check that would
# have caught LIMIT_* being exported by config.env and read by nothing.
if [ "${PLAN_ONLY:-0}" = "1" ]; then
  for T in ${TASKS//,/ }; do
    P="$(plan_for "$T")"
    if [ "$P" = "samples" ]; then
      N="$(samples_json_for "$T" | python3 -c 'import json,sys;print(sum(len(v) for v in json.load(sys.stdin).values()))')" || N="ERR"
      echo "PLAN ${T}=samples:${N}"
    else
      echo "PLAN ${T}=${P}"
    fi
  done
  exit 0
fi

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
# the installed lm-eval 0.4.12 task configs (2026-07-27):
#   mmlu_pro custom-extract : "answer is \(?([A-J])\)?"
#   gpqa strict-match       : "(?<=The answer is )(.*)(?=.)"
#   minerva_math500         : math_verify (format-free equivalence) is primary
# aime25 and ifeval do NOT take this line — see instruction_for() above.
# A "#### <answer>" instruction (an earlier attempt) parses on NONE of them and
# would have scored obedience to our own instruction as failure.
FORMAT_INSTRUCTION="${FORMAT_INSTRUCTION:-End your reply with a final line of the form: The answer is <answer>.}"

MODEL_ARGS="model=${MODEL},base_url=${URL}/v1/chat/completions,num_concurrent=${NUM_CONCURRENT:-4},max_retries=3,tokenized_requests=False,timeout=7200,seed=${SEED}"
PLAN=""   # "task=plan;..." — recorded so the manifest states the realized plan

for T in ${TASKS//,/ }; do
  P="$(plan_for "$T")"; [ "$P" = "ABORT" ] && exit 1
  INSTR="$(instruction_for "$T")"
  EXTRA=()
  case "$P" in
    full)    ;;                                  # whole dataset, by name
    samples) [ -f "${SAMPLES_FILE:-/nonexistent}" ] || {
               echo "[lm_eval] FATAL: plan says 'samples' but SAMPLES_FILE=${SAMPLES_FILE:-<unset>} is missing" >&2
               exit 1; }
             EXTRA+=(--samples "$(samples_json_for "$T")") || exit 1;;
    *)       EXTRA+=(--limit "$P");;             # integer: smoke path (FIRST-N, not a sample)
  esac
  # ifeval scores per-item instructions -> its INSTR is empty -> no system flag.
  [ -n "$INSTR" ] && EXTRA+=(--system_instruction "$INSTR")
  echo "[lm_eval] $NAME task=$T plan=$P instr=${INSTR:-<none>}"
  "$LMEVAL_BIN" "${LMEVAL_RUN[@]}" --model local-chat-completions \
    --model_args "$MODEL_ARGS" \
    --tasks "$T" \
    --num_fewshot "$FEWSHOT" \
    --apply_chat_template \
    --gen_kwargs "temperature=${TEMP},top_p=${TOP_P},max_gen_toks=${GEN_TOKS}" \
    --seed "$SEED" --batch_size 1 --log_samples \
    --output_path "$OUT" \
    "${EXTRA[@]}"
  PLAN="${PLAN}${T}=${P};"
done

python3 - "$OUT" "$TASKS" "$PLAN" "${SAMPLES_FILE:-}" <<'PY'
import hashlib, json, sys
out, tasks, plan, sfile = sys.argv[1], sys.argv[2].split(","), sys.argv[3], sys.argv[4]
limits = dict(kv.split("=", 1) for kv in plan.strip(";").split(";") if kv)
sha = hashlib.sha256(open(sfile, "rb").read()).hexdigest() if sfile else None
json.dump({"tasks_run": tasks,
  "per_task_plan": limits,
  "samples_file_sha256": sha,
  "plan_semantics": "full=whole dataset; samples=frozen seeded doc indices (--samples); int=--limit smoke (FIRST-N)",
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
