"""Generate a synthetic SFT dataset for fine-tuning the local model, targeting
the 8 official evaluation categories (Participant Guide): factual knowledge,
mathematical reasoning, sentiment classification, text summarisation, named
entity recognition, code debugging, logical/deductive reasoning, code
generation.

Two-pass pipeline, both passes using a strong teacher model over the
Fireworks API (the personal dev key only reaches non-judge models; the
teacher is a stand-in, never shipped or referenced in production code):

  Pass 1 (task generation): ask the teacher for NEW gradeable tasks per
    category, styled on the Participant Guide's own practice examples, with
    a machine-checkable `grader`/`expected` pair in the practice.jsonl
    schema. Then filter: schema-valid, code tasks actually pass their own
    test_code, decontaminated against eval/tasks/practice.jsonl (near-dupes
    dropped), deduplicated against each other.

  Pass 2 (answer generation): answer each surviving task using the EXACT
    production local system prompt (profile_for(classify(text)).local_style)
    so the SFT target matches what llama-server will be asked to produce at
    inference time. Keep only answers that pass the task's own grader --
    training on a wrong answer would teach the model to be confidently
    wrong.

Output: train/data/synthetic_tasks.jsonl (audit trail, one raw generated
task per line) and train/data/sft.jsonl (one {"messages": [...]} SFT example
per line, ready for a chat-template SFT script).

Usage:
  python eval/gen_synthetic.py --out train/data [--per-category N]

Requires FIREWORKS_API_KEY in the environment (and optionally
FIREWORKS_BASE_URL / CHEAPROUTE_REMOTE_MODEL to pick the teacher model --
defaults to gpt-oss-120b, the only strong model reachable by the personal
dev key used for local validation this session).
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))          # for graders
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from graders import _run_python, grade  # noqa: E402

from cheaproute.confidence import extract_final  # noqa: E402
from cheaproute.tasktype import classify, extract_answer, profile_for  # noqa: E402

TEACHER_MODEL = os.environ.get("CHEAPROUTE_REMOTE_MODEL",
                               "accounts/fireworks/models/gpt-oss-120b")
BASE_URL = os.environ.get("FIREWORKS_BASE_URL",
                          "https://api.fireworks.ai/inference/v1")

# category -> (grader kind, target count, teacher-facing description +
# schema instructions + style exemplars drawn from the Participant Guide's
# own practice tasks so generated items match the real distribution)
_FENCE_STRIP = re.compile(r"^```[a-zA-Z0-9_+-]*\n|\n```$", re.MULTILINE)

CATEGORIES: dict[str, dict] = {
    "factual": {
        "count": 400,
        "grader": "contains",
        "instructions": (
            "Generate factual-knowledge questions (explaining concepts, "
            "definitions, how things work). Some should be multi-part "
            "(e.g. 'What is the capital of Australia, and what body of "
            "water is it near?'). `expected` is a list of short strings "
            "that must ALL appear in a correct answer (one per part)."
        ),
        "example_schema": '{"type": "factual", "task": "...", '
                          '"expected": ["canberra", "lake burley griffin"], '
                          '"grader": "contains_all"}',
        "grader_override": "contains_all",
    },
    "math": {
        "count": 400,
        "grader": "numeric",
        "instructions": (
            "Generate mathematical-reasoning word problems: multi-step "
            "arithmetic, percentages, projections (e.g. 'A store has 240 "
            "items. It sells 15% on Monday and 60 more on Tuesday. How many "
            "items remain?'). `expected` is the final numeric answer as a "
            "string."
        ),
        "example_schema": '{"type": "math", "task": "...", '
                          '"expected": "144", "grader": "numeric"}',
    },
    "sentiment": {
        "count": 300,
        "grader": "contains",
        "instructions": (
            "Generate sentiment-classification tasks. Each `task` MUST be "
            "phrased exactly like: \"Classify the sentiment of this review "
            "as positive or negative: '<review text>'\" -- invent a new "
            "review/statement each time (a review can mention one minor "
            "downside/upside as long as the OVERALL sentiment is "
            "unambiguous -- do not invent genuinely 50/50 mixed cases, "
            "since the classifier only supports a binary label). "
            "`expected` is the single correct label word, either "
            "\"positive\" or \"negative\"."
        ),
        "example_schema": '{"type": "sentiment", "task": "Classify the '
                          'sentiment of this review as positive or '
                          'negative: \'...\'", "expected": "positive", '
                          '"grader": "contains"}',
    },
    "summarization": {
        "count": 300,
        "grader": "word_limit",
        "instructions": (
            "Generate summarization tasks: a short passage (3-6 sentences, "
            "invent realistic content) plus an instruction to summarize it "
            "in exactly one sentence or under N words. Give the model "
            "realistic room to comply: if the instruction is a plain "
            "'in one sentence' with no explicit word count, set "
            "`max_words` to 30-40 (a real one-sentence summary of a few "
            "facts routinely runs 25-35 words); only use a tight `max_words` "
            "like 15-22 when the task text ITSELF states an explicit count "
            "(e.g. 'in 20 words or fewer'), matching that count plus a "
            "couple words of slack. `expected` is {\"max_words\": N, "
            "\"any_of\": [2-4 keywords a correct summary would very likely "
            "include]}."
        ),
        "example_schema": '{"type": "summarization", "task": "Summarize the '
                          'following in exactly one sentence: <passage>", '
                          '"expected": {"max_words": 25, '
                          '"any_of": ["keyword1", "keyword2"]}, '
                          '"grader": "word_limit"}',
    },
    "ner": {
        "count": 280,
        "grader": "contains_all",
        "instructions": (
            "Generate named-entity-recognition tasks: a sentence with "
            "person/organization/location/date entities, asking to extract "
            "and label them (e.g. 'Extract all named entities and their "
            "types from: Maria Sanchez joined Fireworks AI in Berlin last "
            "March.'). `expected` is a list of the entity strings that must "
            "all appear in a correct answer."
        ),
        "example_schema": '{"type": "ner", "task": "...", '
                          '"expected": ["maria sanchez", "fireworks ai", '
                          '"berlin"], "grader": "contains_all"}',
    },
    "code_debug": {
        "count": 400,
        "batch_size": 6,
        "max_tokens": 6000,
        "grader": "py_exec",
        "instructions": (
            "Generate Python code-debugging tasks: a short function with a "
            "clear, single bug and a description of the intended behavior "
            "(e.g. 'This function should return the max of a list but has "
            "a bug: def get_max(nums): return nums[0]. Find and fix it.'). "
            "`expected` is the CORRECTED function as a fenced python code "
            "block (this is the reference fix, used to verify test_code, "
            "not shown to the model being trained). `test_code` is 2-3 "
            "assert statements that the corrected function must pass."
        ),
        "example_schema": '{"type": "code_debug", "task": "...", '
                          '"expected": "```python\\ndef get_max(nums):\\n'
                          '    return max(nums)\\n```", '
                          '"test_code": "assert get_max([1,5,3]) == 5\\n'
                          'assert get_max([2,2]) == 2", "grader": "py_exec"}',
    },
    "logic": {
        "count": 1200,
        "batch_size": 10,
        "max_tokens": 6000,
        "temperature": 0.55,
        "grader": "contains",
        "instructions": (
            "Generate logical/deductive-reasoning puzzles where all given "
            "constraints must be satisfied to reach a unique answer (e.g. "
            "'Three friends, Sam, Jo, and Lee, each own a different pet: "
            "cat, dog, bird. Sam does not own the bird. Jo owns the dog. "
            "Who owns the cat?'). `expected` is the short final answer."
        ),
        "example_schema": '{"type": "logic", "task": "...", '
                          '"expected": "sam", "grader": "contains"}',
    },
    "code_gen": {
        "count": 400,
        "batch_size": 6,
        "max_tokens": 6000,
        "grader": "py_exec",
        "instructions": (
            "Generate code-generation specs: a clear description of a "
            "Python function to write, including edge cases to handle "
            "(e.g. 'Write a Python function that returns the second-"
            "largest number in a list, handling duplicates correctly.'). "
            "`expected` is a REFERENCE correct implementation as a fenced "
            "python code block. `test_code` is 3-4 assert statements "
            "covering the spec including edge cases."
        ),
        "example_schema": '{"type": "code_gen", "task": "...", '
                          '"expected": "```python\\ndef f(...):\\n    ...\\n'
                          '```", "test_code": "assert f(...) == ...", '
                          '"grader": "py_exec"}',
    },
}

_GEN_SYSTEM = (
    "You generate training data for an AI evaluation benchmark. Output "
    "ONLY a JSON array of task objects, no commentary, no markdown fences. "
    "Every task must be self-contained, unambiguous, and have a single "
    "objectively correct answer. Do not reuse the style examples verbatim "
    "-- invent new content (different numbers, names, topics, code)."
)

_ANSWER_STYLE_RIDER = {
    "short": (" At most 1-2 brief reasoning sentences, then the final line "
             "exactly as instructed."),
    "code": " Return only the fenced code block, no commentary.",
    "freeform": " Output only the requested text, nothing else.",
}


def _fireworks_chat(api_key: str, system: str, user: str,
                    max_tokens: int, temperature: float) -> str:
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        json={
            "model": TEACHER_MODEL,
            "messages": [{"role": "system", "content": system},
                        {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"].get("content") or ""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_STRIP.sub("", text).strip()
    return text


def _generate_batch(api_key: str, category: str, spec: dict,
                    batch_size: int, start_idx: int) -> list[dict]:
    prompt = (
        f"{spec['instructions']}\n\nEach object must match this schema "
        f"exactly (extra/missing keys not allowed): {spec['example_schema']}"
        f"\n\nGenerate a JSON array of exactly {batch_size} NEW, DISTINCT "
        f"tasks of this kind. Return ONLY the JSON array."
    )
    try:
        raw = _fireworks_chat(api_key, _GEN_SYSTEM, prompt,
                              max_tokens=spec.get("max_tokens", 4000),
                              temperature=spec.get("temperature", 0.9))
        items = json.loads(_strip_json_fences(raw))
        if not isinstance(items, list):
            return []
    except Exception as exc:
        print(f"  [{category}] batch@{start_idx} generation failed: {exc}",
              file=sys.stderr)
        return []

    out = []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or "task" not in item or "expected" not in item:
            continue
        item["id"] = f"syn-{category}-{start_idx + i}"
        item["type"] = category
        item.setdefault("grader", spec.get("grader_override", spec["grader"]))
        out.append(item)
    return out


def _validate_and_filter(items: list[dict], practice: list[dict]) -> list[dict]:
    """Schema/execution validation + decontamination + dedup."""
    practice_norm = [_norm(t["task"]) for t in practice]
    seen: list[str] = []
    kept = []
    for item in items:
        task_text = (item.get("task") or "").strip()
        if not task_text or len(task_text) < 8:
            continue
        expected = item.get("expected")
        if expected is None or expected == "":
            continue
        grader = item.get("grader", "exact")

        if grader == "py_exec":
            test_code = item.get("test_code", "")
            if not test_code:
                continue
            # verify the reference `expected` implementation itself passes
            # its own test_code before trusting it as ground truth
            if not _run_python(str(expected), test_code):
                continue

        norm = _norm(task_text)
        if any(norm == p or _similar(norm, p) for p in practice_norm):
            continue  # near-duplicate of a held-out practice task
        if any(_similar(norm, s) for s in seen):
            continue  # near-duplicate within this synthetic batch
        seen.append(norm)
        kept.append(item)
    return kept


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _similar(a: str, b: str, threshold: float = 0.93) -> bool:
    """Near-duplicate check. difflib's ratio conflates "shares a template"
    with "shares content" -- two distinct-fact questions built on the same
    template ("capital of France?" / "capital of Germany?") can score
    ~0.88, so the threshold sits above that band and relies on the substring
    check to still catch true near-verbatim duplicates/paraphrases."""
    if a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() > threshold


def _answer_style_rider(ttype: str) -> str:
    if ttype in ("code_debug", "code_gen"):
        return _ANSWER_STYLE_RIDER["code"]
    prof = profile_for(ttype)
    return _ANSWER_STYLE_RIDER["freeform" if prof.freeform else "short"]


def _answer_task(api_key: str, item: dict) -> dict | None:
    """Pass 2: answer with the production local system prompt, keep only
    grader-passing answers."""
    task_text = item["task"]
    ttype = classify(task_text)
    prof = profile_for(ttype)
    system = f"You are a careful assistant. {prof.local_style}" + \
        _answer_style_rider(ttype)
    try:
        raw_answer = _fireworks_chat(api_key, system, task_text,
                                     max_tokens=max(prof.local_max_tokens, 256),
                                     temperature=0.3)
    except Exception:
        return None
    if not raw_answer.strip():
        return None

    extracted = extract_answer(ttype, raw_answer, extract_final)
    grade_task = dict(item)
    grade_task["grader"] = item.get("grader", "exact")
    if item["type"] == "summarization" and isinstance(item.get("expected"), dict):
        # Pass-1's `any_of` keywords are the teacher's own guess at pass-1
        # time about likely phrasing; a differently-worded but equally
        # correct pass-2 summary shouldn't be rejected as an SFT example
        # for missing that exact word. Word-limit compliance still gates.
        grade_task["expected"] = {"max_words": item["expected"].get("max_words")}
    if not grade(grade_task, extracted):
        return None

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": task_text},
            {"role": "assistant", "content": raw_answer.strip()},
        ],
        "meta": {"id": item["id"], "type": item["type"],
                "classified_as": ttype},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="train/data")
    ap.add_argument("--per-category", type=int, default=None,
                    help="override the per-category generation target")
    ap.add_argument("--gen-batch-size", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--practice", default="eval/tasks/practice.jsonl")
    ap.add_argument("--categories", default=None,
                    help="comma-separated subset of CATEGORIES to run "
                         "(default: all) -- lets a single category be "
                         "(re)run without repeating the whole pipeline")
    ap.add_argument("--append", action="store_true",
                    help="append to existing synthetic_tasks.jsonl/sft.jsonl "
                         "instead of overwriting (default: overwrite)")
    args = ap.parse_args()

    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    if not api_key:
        print("FIREWORKS_API_KEY is not set", file=sys.stderr)
        return 1

    with open(args.practice, encoding="utf-8-sig") as f:
        practice = [json.loads(line) for line in f if line.strip()]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = CATEGORIES
    if args.categories:
        wanted = [c.strip() for c in args.categories.split(",") if c.strip()]
        categories = {c: CATEGORIES[c] for c in wanted}

    raw_path = out_dir / "synthetic_tasks.jsonl"
    sft_path = out_dir / "sft.jsonl"
    file_mode = "a" if args.append else "w"

    # ---- Pass 1: generate + filter, streamed to disk per category -----
    # Written incrementally (not batched into one big end-of-run write) so
    # an interrupted run (background-task teardown, pod timeout, etc.)
    # keeps everything completed so far instead of losing the whole pass.
    all_raw: list[dict] = []
    with open(raw_path, file_mode, encoding="utf-8") as raw_f:
        for category, spec in categories.items():
            target = args.per_category or spec["count"]
            batch_size = spec.get("batch_size", args.gen_batch_size)
            n_batches = max(1, -(-target // batch_size))  # ceil
            print(f"[gen] {category}: requesting {n_batches} batches "
                  f"({batch_size} each, target {target})", file=sys.stderr)
            batch_items: list[dict] = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = [pool.submit(_generate_batch, api_key, category, spec,
                                   batch_size, b * batch_size)
                       for b in range(n_batches)]
                for fut in as_completed(futs):
                    batch_items.extend(fut.result())
            kept = _validate_and_filter(batch_items, practice)
            print(f"[gen] {category}: {len(batch_items)} raw -> {len(kept)} "
                  f"after validation/decontamination", file=sys.stderr)
            for item in kept:
                raw_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            raw_f.flush()
            all_raw.extend(kept)
    print(f"[gen] wrote {len(all_raw)} candidate tasks -> {raw_path}",
          file=sys.stderr)

    # ---- Pass 2: answer with production prompts, keep grader-passing --
    # Also streamed: each accepted example is flushed as soon as it's ready.
    by_type: dict[str, int] = {}
    kept_count = 0
    with open(sft_path, file_mode, encoding="utf-8") as sft_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_answer_task, api_key, item) for item in all_raw]
            for i, fut in enumerate(as_completed(futs)):
                result = fut.result()
                if result:
                    sft_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    sft_f.flush()
                    kept_count += 1
                    t = result["meta"]["type"]
                    by_type[t] = by_type.get(t, 0) + 1
                if (i + 1) % 200 == 0:
                    print(f"[answer] {i + 1}/{len(all_raw)} processed, "
                          f"{kept_count} kept so far", file=sys.stderr)

    print(f"\n=== {kept_count} SFT examples -> {sft_path} ===", file=sys.stderr)
    for t in sorted(by_type):
        print(f"  {t:<14} {by_type[t]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
