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
    ;;
  solar)
    MODEL="${2:-$SOLAR_MODEL}"; PORT="${3:-8082}"
    # If Solar's MLX model class lives in a separate fork, put it on PYTHONPATH:
    export PYTHONPATH="${SOLAR_FORK:-}${SOLAR_FORK:+:}${PYTHONPATH:-}"
    ;;
  *) echo "usage: serve.sh motif|solar [model_dir] [port]"; exit 1;;
esac

echo "[serve] $which  model=$MODEL  port=$PORT"
echo "[serve] PYTHONPATH=${PYTHONPATH:-<system mlx_lm>}"
exec python3 -m mlx_lm.server \
  --model "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code \
  --log-level INFO
