#!/usr/bin/env bash
# One-time environment setup on the AMD hackathon Jupyter pod
# (notebooks.amd.com/hackathon: ROCm 7.2 + vLLM 0.16 + PyTorch 2.9 preinstalled).
# Run from a terminal inside the pod, from the repo root after cloning/copying
# this repo there.
set -euo pipefail

echo "[pod_setup] verifying preinstalled PyTorch sees the GPU..."
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'; print(torch.__version__, torch.cuda.get_device_name(0))"

echo "[pod_setup] installing fine-tuning stack (bf16 LoRA only -- no bitsandbytes/QLoRA on ROCm)..."
pip install --user --upgrade \
    "transformers>=4.53" "peft>=0.15" "trl>=0.19" \
    datasets accelerate sentencepiece protobuf "huggingface_hub[cli]"

echo "[pod_setup] building llama.cpp (CPU build -- mirrors the judge's CPU-only runtime)..."
if [ ! -d "$HOME/llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$HOME/llama.cpp"
fi
cmake -B "$HOME/llama.cpp/build" -S "$HOME/llama.cpp" -DLLAMA_CURL=OFF
cmake --build "$HOME/llama.cpp/build" -j --target llama-quantize llama-server
pip install --user -r "$HOME/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"

echo "[pod_setup] fetching baseline GGUF for the gate-eval comparison..."
mkdir -p "$HOME/models"
if [ ! -s "$HOME/models/baseline.gguf" ]; then
    curl -fL --retry 3 --retry-all-errors -C - \
        -o "$HOME/models/baseline.gguf" \
        "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf"
fi

echo "[pod_setup] done. Next: scp train/data/sft.jsonl here, then run sft_lora.py"
