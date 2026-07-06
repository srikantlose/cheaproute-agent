#!/usr/bin/env bash
# CheapRoute container entrypoint:
#   1. start llama-server (CPU) with the baked local GGUF
#   2. wait until healthy — bounded so total readiness stays under the 60s cap
#   3. exec the routing agent (batch adapter: /input/tasks.json ->
#      /output/results.json, then exit 0)
#
# Env knobs: LOCAL_MODEL_PATH, LLAMA_PORT, LLAMA_CTX, LLAMA_PARALLEL,
#            LLAMA_THREADS, LLAMA_EXTRA_ARGS, SKIP_LOCAL=1 (no local model),
#            plus harness vars (FIREWORKS_API_KEY/BASE_URL, ALLOWED_MODELS)
#            and CHEAPROUTE_* overrides.
set -u

MODEL_PATH="${LOCAL_MODEL_PATH:-/models/local.gguf}"
PORT="${LLAMA_PORT:-8000}"

if [ "${SKIP_LOCAL:-0}" != "1" ] && [ -f "$MODEL_PATH" ]; then
    echo "[entrypoint] starting llama-server (${MODEL_PATH}) on :${PORT}" >&2
    /opt/llama/llama-server -m "$MODEL_PATH" \
        --host 127.0.0.1 --port "$PORT" \
        -c "${LLAMA_CTX:-8192}" \
        --parallel "${LLAMA_PARALLEL:-4}" \
        -t "${LLAMA_THREADS:-$(nproc)}" \
        ${LLAMA_EXTRA_ARGS:-} >&2 &

    # <=50 x 1s: leaves headroom inside the 60-second readiness requirement.
    ready=0
    for _ in $(seq 1 50); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 1
    done
    if [ "$ready" = "1" ]; then
        echo "[entrypoint] llama-server is healthy" >&2
    else
        # Do NOT exit: the router degrades gracefully by escalating remotely.
        echo "[entrypoint] WARNING: llama-server not healthy yet; continuing" >&2
    fi
fi

export CHEAPROUTE_LOCAL_BASE_URL="http://127.0.0.1:${PORT}/v1"

exec python -m cheaproute --adapter batch "$@"
