"""LoRA SFT of the local model on the synthetic distillation dataset
(train/data/sft.jsonl, produced by eval/gen_synthetic.py). Runs on the AMD
hackathon Jupyter pod (ROCm + PyTorch preinstalled) or the DO MI300X droplet
fallback. bf16 LoRA only -- no bitsandbytes/QLoRA (unreliable on ROCm at the
time of writing), and the base models here are small enough that bf16 full
weights + a LoRA adapter fit trivially on an MI300X.

Usage:
  python train/sft_lora.py [--base-model MODEL_ID] [--data train/data/sft.jsonl]
                           [--out train/checkpoints/lora] [--epochs 2] [--lr 1e-4]

BASE_MODEL env var (or --base-model) picks the target: default
google/gemma-3-1b-it (ungated mirror unsloth/gemma-3-1b-it if the HF token
hasn't accepted Google's license). Swap for Qwen2.5-3B-Instruct or
Llama-3.2-3B-Instruct if the D3.5 model-size decision picked one of those
instead -- target_modules below covers the standard attention+MLP proj
names shared by all three architectures; only gemma-3-4b (multimodal) needs
the vision-excluding regex, and this script isn't intended for the 4B.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments)
from trl import SFTConfig, SFTTrainer

try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    DataCollatorForCompletionOnlyLM = None

# Response templates per chat-template family, used to mask the loss to the
# assistant turn only (completion-only loss). If a model's tokenizer doesn't
# emit one of these markers verbatim, fall back to full-sequence loss --
# noted in the training log rather than silently mis-masking.
_RESPONSE_TEMPLATES = {
    "gemma": "<start_of_turn>model\n",
    "qwen": "<|im_start|>assistant\n",
    "llama": "<|start_header_id|>assistant<|end_header_id|>\n\n",
}


def _guess_family(model_id: str) -> str:
    m = model_id.lower()
    if "gemma" in m:
        return "gemma"
    if "qwen" in m:
        return "qwen"
    if "llama" in m:
        return "llama"
    return "gemma"


def load_sft_dataset(path: Path) -> Dataset:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def render_example(tokenizer, messages: list[dict]) -> str:
    """Render one SFT example via the model's own chat template so training
    text matches what llama-server will render from the same [system, user]
    turns at inference time (verified separately in merge_and_convert.sh's
    /apply-template parity check)."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        # Some chat templates (gemma included, in some versions) reject a
        # bare "system" role. Fold it into the first user turn instead --
        # this must match how llama.cpp's built-in gemma3 template handles
        # a system message sent over /v1/chat/completions.
        sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        rest = [m for m in messages if m["role"] != "system"]
        if rest and rest[0]["role"] == "user":
            rest[0] = {"role": "user",
                      "content": f"{sys_msg}\n\n{rest[0]['content']}"}
        return tokenizer.apply_chat_template(rest, tokenize=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=os.environ.get(
        "BASE_MODEL", "google/gemma-3-1b-it"))
    ap.add_argument("--data", default="train/data/sft.jsonl")
    ap.add_argument("--out", default="train/checkpoints/lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--eval-holdout", type=int, default=100)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no GPU visible to torch -- check the pod's ROCm setup",
              file=sys.stderr)
        return 1

    print(f"[sft] base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="eager")

    family = _guess_family(args.base_model)
    response_template = _RESPONSE_TEMPLATES[family]
    if response_template not in tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
            tokenize=False):
        print(f"[sft] WARNING: response template {response_template!r} not "
              f"found in this tokenizer's rendering -- falling back to "
              f"full-sequence loss (completion-only masking skipped)",
              file=sys.stderr)
        response_template = None

    dataset = load_sft_dataset(Path(args.data))
    dataset = dataset.map(lambda ex: {
        "text": render_example(tokenizer, ex["messages"])})
    dataset = dataset.shuffle(seed=42)
    n_eval = min(args.eval_holdout, max(1, len(dataset) // 20))
    eval_ds = dataset.select(range(n_eval))
    train_ds = dataset.select(range(n_eval, len(dataset)))
    print(f"[sft] {len(train_ds)} train / {len(eval_ds)} eval examples")

    # gemma-3-4b is multimodal (has a vision tower); exclude it explicitly
    # so LoRA only ever touches language-model attention/MLP projections.
    # 1b/3b text-only models have no vision tower, so this regex is a no-op
    # for them but kept for safety if BASE_MODEL is ever pointed at -4b-it.
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=r"^(?!.*vision).*\.(q_proj|k_proj|v_proj|o_proj|"
                       r"gate_proj|up_proj|down_proj)$",
    )
    model = get_peft_model(model, lora_config)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert not any("vision" in n for n in trainable), \
        "LoRA touched a vision-tower parameter -- target_modules regex is wrong"
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[sft] trainable params: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.2f}%)")

    collator = None
    if response_template and DataCollatorForCompletionOnlyLM is not None:
        collator = DataCollatorForCompletionOnlyLM(
            response_template, tokenizer=tokenizer)

    sft_config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model, args=sft_config,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator, processing_class=tokenizer,
    )
    trainer.train()

    print("[sft] training complete, saving adapter...")
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    print("[sft] mid/post-train sanity generation on fixed prompts:")
    _sanity_check(model, tokenizer)
    return 0


_SANITY_PROMPTS = [
    ("math", "What is 17 + 25?"),
    ("mc", "Which is the largest planet? A) Earth B) Jupiter C) Mars D) Venus. "
          "Answer with the letter."),
    ("sentiment", "Classify the sentiment of this review as positive or "
                 "negative: 'Great product, highly recommend.'"),
    ("summarization", "Summarize the following in exactly one sentence: "
                      "The quick brown fox jumps over the lazy dog near "
                      "the riverbank every morning at dawn."),
    ("ner", "Extract all person names and locations from: 'Alice went to "
           "Rome to meet Bob.'"),
    ("code_debug", "This function has a bug, fix it:\n\ndef add(a, b):\n"
                   "    return a - b"),
    ("logic", "What is the next number in the sequence 2, 4, 8, 16?"),
    ("code_gen", "Write a Python function that returns the square of a "
                "number."),
]


def _sanity_check(model, tokenizer) -> None:
    model.eval()
    for label, prompt in _SANITY_PROMPTS:
        messages = [
            {"role": "system", "content": "You are a careful assistant. "
                                          "Solve the task. Think briefly if "
                                          "needed, then give the complete "
                                          "final answer on the last line in "
                                          "the form:\nAnswer: <complete "
                                          "final answer>"},
            {"role": "user", "content": prompt},
        ]
        text = render_example(tokenizer, messages)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        has_marker = ("answer:" in gen.lower()) or ("```" in gen)
        flag = "OK" if has_marker else "MISSING ANSWER MARKER"
        print(f"  [{label}] [{flag}] {gen.strip()[:200]!r}")
    model.train()


if __name__ == "__main__":
    sys.exit(main())
