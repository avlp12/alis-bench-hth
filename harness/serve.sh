#!/usr/bin/env bash
# serve.sh <motif|solar> [model_dir] [port]
# Serve an MLX build via mlx_lm.server (OpenAI-compatible /v1/chat/completions).
# Motif needs the @motif fork on PYTHONPATH (its model class is not in upstream mlx-lm).
# Solar needs whatever fork registers its model class (set SOLAR_FORK if separate).
#
# Run one of these on EACH box:
#   Gesicht $  ./serve.sh motif                 # -> :8081
#   Epsilon $  ./serve.sh solar                 # -> :8082
#
# Sampling is NOT fixed here on purpose — the eval clients set temp/seed per request
# so the server default never biases a task.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/config.env"
which="${1:?usage: serve.sh motif|solar [model_dir] [port]}"

case "$which" in
  motif)
    MODEL="${2:-$MOTIF_MODEL}"; PORT="${3:-8081}"
    export PYTHONPATH="$MLXLM_FORK:${PYTHONPATH:-}"    # @motif model class
    PY="${MOTIF_PYTHON:-python3}"; ADAPTER="${MOTIF_ADAPTER:-}"
    ;;
  solar)
    MODEL="${2:-$SOLAR_MODEL}"; PORT="${3:-8082}"
    # Solar runs from Kimi's venv (editable fork with the solar_open2 class);
    # do NOT put its fork on PYTHONPATH as well — the venv already resolves it.
    PY="${SOLAR_PYTHON:-python3}"; ADAPTER="${SOLAR_ADAPTER:-}"
    ;;
  *) echo "usage: serve.sh motif|solar [model_dir] [port]"; exit 1;;
esac

echo "[serve] $which  model=$MODEL  port=$PORT"
echo "[serve] python=$PY  adapter=${ADAPTER:-<none>}"
exec "$PY" -m mlx_lm.server \
  --model "$MODEL" ${ADAPTER:+--adapter-path "$ADAPTER"} \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code \
  --log-level INFO
