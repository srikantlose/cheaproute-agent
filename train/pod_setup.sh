#!/usr/bin/env bash
# One-time environment setup on the AMD hackathon Jupyter pod
# (notebooks.amd.com/hackathon). The instance we drew ships ROCm 5.7 +
# PyTorch 2.3.1+rocm5.7 preinstalled (NOT the 2.9 the docs advertise), and
# that pinned torch is the constraint everything below bends around. Run from
# a terminal inside the pod, from the repo root after cloning/copying the repo.
set -euo pipefail

echo "[pod_setup] verifying preinstalled PyTorch sees the GPU..."
# If this reports a +cpu build (a plain `pip install torch` elsewhere can
# clobber the ROCm wheel), reinstall the ROCm build before continuing:
#   pip install torch --index-url https://download.pytorch.org/whl/rocm5.7
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'; print(torch.__version__, torch.cuda.get_device_name(0))"

echo "[pod_setup] installing fine-tuning stack (bf16 LoRA only -- no bitsandbytes/QLoRA on ROCm)..."
# No trl. trl 1.x hard-requires transformers>=4.56.2, which in turn requires
# torch>=2.4 -- unsatisfiable against this pod's pinned torch 2.3.1, and every
# older trl that would install just shifts the SFTConfig/collator API break
# somewhere else. So train/sft_lora.py drives transformers' plain Trainer +
# peft directly (the completion-only masking trl used to provide lives in the
# script's own collator).
#
# transformers is pinned into a narrow band: gemma-3 support only landed in
# 4.50 (older versions don't know model_type "gemma3" and fail to load the
# base model), and 4.56+ disables torch on anything below 2.4. 4.53.3 sits in
# the overlap -- knows gemma-3, still runs on torch 2.3.1 -- with peft on top.
pip install --user --upgrade \
    "transformers==4.53.3" "peft>=0.15" \
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
