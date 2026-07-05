#!/usr/bin/env bash
# CheapRoute container entrypoint:
#   1. start vLLM (ROCm) serving the local model on localhost
#   2. wait until it reports healthy (bounded wait, don't hang forever)
#   3. exec the routing agent with whichever adapter CHEAPROUTE_ADAPTER selects
#
# Env knobs: LOCAL_MODEL, VLLM_PORT, VLLM_MAX_LEN, VLLM_EXTRA_ARGS,
#            SKIP_VLLM=1 (dev/testing without a GPU: mock or external local),
#            FIREWORKS_API_KEY (required for remote escalation),
#            CHEAPROUTE_* overrides (see config.py).
set -u

LOCAL_MODEL="${LOCAL_MODEL:-google/gemma-3-4b-it}"
PORT="${VLLM_PORT:-8000}"

if [ "${SKIP_VLLM:-0}" != "1" ]; then
    echo "[entrypoint] starting vLLM with ${LOCAL_MODEL} on :${PORT}" >&2
    vllm serve "$LOCAL_MODEL" \
        --host 127.0.0.1 --port "$PORT" \
        --max-model-len "${VLLM_MAX_LEN:-4096}" \
        ${VLLM_EXTRA_ARGS:-} >&2 &

    # Bounded health wait: ~6 minutes max (large first-load on cold cache).
    ready=0
    for _ in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 2
    done
    if [ "$ready" = "1" ]; then
        echo "[entrypoint] vLLM is healthy" >&2
    else
        # Do NOT exit: the router degrades gracefully by escalating remotely.
        echo "[entrypoint] WARNING: vLLM never became healthy; continuing" >&2
    fi
fi

export CHEAPROUTE_LOCAL_BASE_URL="http://127.0.0.1:${PORT}/v1"
export CHEAPROUTE_LOCAL_MODEL="$LOCAL_MODEL"

exec python -m cheaproute "$@"
