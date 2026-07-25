#!/usr/bin/env bash
# run_kl.sh <motif|solar>
# Per-model quantization DAMAGE, measured as KL(own-8bit-ref || quant) on the
# fixed EN/code/KO slices via alis-dwq. This is the number that SEPARATES
# "quant hurt model X" from "model X is a weaker base model" — run it for BOTH
# so a low benchmark score can be attributed correctly.
#
# Loads the MLX model in-process (NOT via the server) — the correct fidelity
# measure. Needs each model's class importable (Motif fork / Solar fork).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/config.env"
NAME="${1:?motif|solar}"
case "$NAME" in
  motif) REF="$MOTIF_REF"; CAND="$MOTIF_MODEL"; export PYTHONPATH="$MLXLM_FORK:${PYTHONPATH:-}";;
  solar) REF="$SOLAR_REF"; CAND="$SOLAR_MODEL"; export PYTHONPATH="${SOLAR_FORK:-}${SOLAR_FORK:+:}${PYTHONPATH:-}";;
  *) echo "usage: run_kl.sh motif|solar"; exit 1;;
esac
KL="$RESULTS_DIR/kl"; mkdir -p "$KL"
# Korean eval slice (frozen, disjoint from calibration)
export ALIS_DWQ_LANG_SLICE="${ALIS_DWQ_LANG_SLICE:-KO:$ALISDWQ_DIR/data/ko.txt}"

cd "$ALISDWQ_DIR"
echo "[kl] $NAME  ref=$REF"
python -m alis_dwq.eval_kld --model "$REF"  --save-ref "$KL/${NAME}_ref.npz"
echo "[kl] $NAME  cand=$CAND  (vs own 8-bit ref)"
python -m alis_dwq.eval_kld --model "$CAND" --ref "$KL/${NAME}_ref.npz" | tee "$KL/${NAME}_kl.txt"
# self-stamp into the current run so the integrity gate sees a fresh KL cell
if [ -f "$RESULTS_DIR/run_manifest.json" ]; then
  python3 "$here/manifest.py" record "kl.$NAME" 0
  python3 "$here/manifest.py" stamp "kl/${NAME}_kl.txt"
  echo "[kl] stamped kl.$NAME into the run"
fi
echo "[kl] $NAME done -> $KL/${NAME}_kl.txt"
