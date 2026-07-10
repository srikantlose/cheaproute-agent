#!/usr/bin/env bash
# Merge the LoRA adapter into the base model, convert to GGUF, quantize, and
# smoke-test the result. Run on the pod after train/sft_lora.py finishes.
#
# Usage: train/merge_and_convert.sh [BASE_MODEL] [LORA_DIR] [OUT_PREFIX] [QUANT]
#   defaults:      google/gemma-3-1b-it   train/checkpoints/lora   $HOME/ft   Q8_0
set -euo pipefail

BASE_MODEL="${1:-${BASE_MODEL:-google/gemma-3-1b-it}}"
LORA_DIR="${2:-train/checkpoints/lora}"
OUT_PREFIX="${3:-$HOME/ft}"
QUANT="${4:-Q8_0}"
MERGED_DIR="$HOME/merged"
LLAMA_DIR="$HOME/llama.cpp"

echo "[merge] base=$BASE_MODEL lora=$LORA_DIR quant=$QUANT"

python - "$BASE_MODEL" "$LORA_DIR" "$MERGED_DIR" <<'PYEOF'
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model, lora_dir, merged_dir = sys.argv[1:4]
print(f"[merge] loading base {base_model} in bf16...")
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
print(f"[merge] loading LoRA adapter from {lora_dir}...")
model = PeftModel.from_pretrained(model, lora_dir)
print("[merge] merging and unloading...")
model = model.merge_and_unload()
model.save_pretrained(merged_dir, safe_serialization=True)

tok = AutoTokenizer.from_pretrained(base_model)
tok.save_pretrained(merged_dir)
print(f"[merge] merged model saved -> {merged_dir}")
PYEOF

echo "[convert] HF -> GGUF (f16)..."
python "$LLAMA_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" \
    --outfile "${OUT_PREFIX}-f16.gguf" --outtype f16

echo "[quantize] f16 -> $QUANT..."
"$LLAMA_DIR/build/bin/llama-quantize" "${OUT_PREFIX}-f16.gguf" \
    "${OUT_PREFIX}-${QUANT}.gguf" "$QUANT"

echo "[smoke] launching llama-server for a sanity check..."
"$LLAMA_DIR/build/bin/llama-server" -m "${OUT_PREFIX}-${QUANT}.gguf" \
    --host 127.0.0.1 --port 8010 -c 8192 --parallel 2 -t "$(nproc)" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 30); do
    curl -sf http://127.0.0.1:8010/health >/dev/null 2>&1 && break
    sleep 1
done

echo "[smoke] chat completion test:"
curl -s http://127.0.0.1:8010/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"local","messages":[{"role":"system","content":"You are a careful assistant. Solve the task. Think briefly if needed, then answer every part of the question, giving the complete final answer on the last line in the form:\nAnswer: <complete final answer>"},{"role":"user","content":"What is 12 + 30?"}],"max_tokens":96,"temperature":0}' \
    | python -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])"

echo "[smoke] template parity check (compare against training-side rendering manually)..."
curl -s http://127.0.0.1:8010/apply-template \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"system","content":"You are a careful assistant. Answer with the final number/result only. Plain text only, no markdown or LaTeX. Last line exactly:\nAnswer: <final answer>"},{"role":"user","content":"What is 12 + 30?"}]}'

echo ""
echo "[done] GGUF at ${OUT_PREFIX}-${QUANT}.gguf -- compare the /apply-template output"
echo "above against tokenizer.apply_chat_template() rendering from sft_lora.py's"
echo "render_example() for the same messages. If they differ materially, fix the"
echo "training-side rendering to match and retrain once before trusting the gate eval."
