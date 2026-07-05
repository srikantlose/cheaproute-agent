# CheapRoute scoring container.
# Base: AMD's official ROCm vLLM image — vLLM + PyTorch(ROCm) preinstalled.
# Verify the freshest tag on https://hub.docker.com/r/rocm/vllm before the
# final build; pin it explicitly for reproducibility.
FROM rocm/vllm:latest

WORKDIR /app

# --- application ---------------------------------------------------------
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY config.yaml ./
RUN pip install --no-cache-dir .

# --- model weights -------------------------------------------------------
# Bake the local model into the image so the scoring environment needs no
# HuggingFace network access at runtime. google/gemma-3-* is a GATED repo:
# accept the license on huggingface.co once, then build with
#   docker build --build-arg HF_TOKEN=hf_xxx -f docker/Dockerfile .
ARG LOCAL_MODEL=google/gemma-3-4b-it
ARG HF_TOKEN=""
ENV LOCAL_MODEL=${LOCAL_MODEL} \
    HF_HOME=/models
RUN if [ -n "$HF_TOKEN" ]; then \
        HF_TOKEN=$HF_TOKEN huggingface-cli download "$LOCAL_MODEL"; \
    else \
        echo "WARNING: HF_TOKEN not set — weights NOT baked, vLLM will try" \
             "to download at runtime (needs network + token in the env)"; \
    fi

# --- runtime defaults (all overridable with -e at docker run) -------------
ENV CHEAPROUTE_LOCAL_BACKEND=openai \
    CHEAPROUTE_REMOTE_BACKEND=fireworks \
    CHEAPROUTE_ADAPTER=stdio

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
