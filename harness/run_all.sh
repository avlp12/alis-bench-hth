#!/usr/bin/env bash
# run_all.sh — FAIL-CLOSED orchestrator. Assumes serve.sh is running for BOTH
# models (Motif on Gesicht:8081, Solar on Epsilon:8082).
#
# NOT `set -e`: we capture each step's exit code EXPLICITLY and record it in the
# manifest, then manifest.py check() fail-closes if any required cell is missing,
# stale (wrong run_id), or produced by a failed step. No silent fresh-vs-stale.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/config.env"
mkdir -p "$RESULTS_DIR"/{lm_eval,hret,judge,throughput,kl}

RUN_ID="$(python3 "$here/manifest.py" init)"
echo "== run_id=$RUN_ID  suite_mode=${SUITE_MODE:-full}  -> $RESULTS_DIR =="

rec(){ python3 "$here/manifest.py" record "$1" "$2"; [ "$2" -eq 0 ] && python3 "$here/manifest.py" stamp "$3" || echo "  [$1] FAILED rc=$2 (recorded, will block aggregation)"; }

echo "== 0. sanity / fairness pre-flight =="
python3 "$here/sanity_probe.py" motif || { echo "motif endpoint not fair — fix first"; exit 1; }
python3 "$here/sanity_probe.py" solar || { echo "solar endpoint not fair — fix first"; exit 1; }

echo "== 1. lm-eval (both parallel) =="
( bash "$here/run_lm_eval.sh" motif ) & P1=$!
( bash "$here/run_lm_eval.sh" solar ) & P2=$!
wait $P1; R1=$?; wait $P2; R2=$?
rec lm_eval.motif "$R1" lm_eval/motif
rec lm_eval.solar "$R2" lm_eval/solar

echo "== 2. HRET (both parallel) =="
( python3 "$here/run_hret.py" motif ) & P3=$!
( python3 "$here/run_hret.py" solar ) & P4=$!
wait $P3; R3=$?; wait $P4; R4=$?
rec hret.motif "$R3" hret/motif
rec hret.solar "$R4" hret/solar

echo "== 3. pairwise judge =="
python3 "$here/judge_pairwise.py" ${LOGICKOR:+--prompts "$LOGICKOR"}; RJ=$?
rec judge "$RJ" judge/pairwise.json

echo "== 4. KL + throughput are separate steps — they self-stamp into this run =="
echo "     KL   (where each build+ref live):  ./run_kl.sh motif ; ./run_kl.sh solar"
echo "     THRU (ONE box, back-to-back):"
echo "        ./serve.sh motif & sleep 30; python3 throughput_probe.py --name motif --url http://localhost:8081; kill %1"
echo "        ./serve.sh solar & sleep 30; python3 throughput_probe.py --name solar --url http://localhost:8082; kill %1"

echo "== 5. integrity gate (fail-closed) =="
if python3 "$here/manifest.py" check; then
  echo "== 6. aggregate =="
  python3 "$here/aggregate.py"
else
  echo ">> Aggregation blocked: run the missing cells above (KL/THRU), then:"
  echo ">>   python3 manifest.py check && python3 aggregate.py"
  echo ">> (or ALLOW_PARTIAL=1 python3 manifest.py check && ALLOW_PARTIAL=1 python3 aggregate.py  for a clearly-marked PARTIAL report)"
  exit 1
fi
