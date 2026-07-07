# CheapRoute Track 1 scoring container (linux/amd64).
#
# Constraints from the Participant Guide this image is built around:
#   - compressed size <= 10GB   (this build: ~3GB)
#   - ready within 60s          (llama-server loads a 2.5GB GGUF in seconds)
#   - 10 min max runtime        (batch adapter enforces its own deadline)
#   - judging VM is linux/amd64, no GPU assumed -> CPU llama.cpp inference
#
# Two submission variants from this one file (see README "Routing modes"):
#   docker build --platform linux/amd64 --build-arg ROUTING_MODE=remote_only \
#     -f docker/Dockerfile -t cheaproute:remote-only .
#   docker build --platform linux/amd64 \
#     -f docker/Dockerfile -t cheaproute:local-first .

# Stage 1: grab the llama-server binary + shared libs from the official image.
FROM ghcr.io/ggml-org/llama.cpp:server AS llama

FROM python:3.12-slim

# llama-server runtime dependencies (OpenMP, curl for health checks/downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libcurl4 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=llama /app /opt/llama
ENV LD_LIBRARY_PATH=/opt/llama

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY config.yaml ./
RUN pip install --no-cache-dir .

# Bake the local Gemma GGUF into the image: the judging environment must not
# need HuggingFace network access. Public, ungated repo — no token required.
ARG LOCAL_GGUF_URL=https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf
RUN mkdir -p /models && curl -fL --retry 3 -o /models/local.gguf "$LOCAL_GGUF_URL"

# Runtime defaults — the harness overrides behaviour via FIREWORKS_API_KEY,
# FIREWORKS_BASE_URL and ALLOWED_MODELS; CHEAPROUTE_* knobs stay tunable.
# ROUTING_MODE bakes the variant: local_first | remote_only.
ARG ROUTING_MODE=local_first
ENV CHEAPROUTE_LOCAL_BACKEND=openai \
    CHEAPROUTE_REMOTE_BACKEND=fireworks \
    CHEAPROUTE_ROUTING_MODE=${ROUTING_MODE} \
    CHEAPROUTE_ADAPTER=batch \
    CHEAPROUTE_LOG_PATH=/output/decisions.jsonl \
    LOCAL_MODEL_PATH=/models/local.gguf

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
