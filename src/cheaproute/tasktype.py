"""Task-type awareness for the 8 evaluation categories.

The judge evaluates 8 capability categories (factual, math, sentiment,
summarisation, NER, code debugging, logic, code generation). Types differ in:
  - how the local/remote models should be prompted,
  - how the final answer is extracted (short-form "Answer: X" vs full text),
  - how many self-consistency samples make sense (agreement is meaningless
    for free-form outputs and CPU samples cost latency),
  - how many remote output tokens to allow (remote tokens ARE the score).

Classification is heuristic (regex/keywords) — zero tokens, instant, and a
wrong guess only means slightly suboptimal prompting, never a wrong pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Ordered checks: first match wins. More specific categories come first.
_CODE_MARKERS = re.compile(r"```|def |class |return |function\s|console\.log|"
                           r"print\(|import |;\s*$|\{.*\}", re.MULTILINE)
_DEBUG_WORDS = re.compile(r"\b(bug|debug|fix|broken|incorrect|error|wrong|"
                          r"not work|doesn'?t work|faulty)\b", re.IGNORECASE)
_CODEGEN_WORDS = re.compile(r"\b(write|implement|create|build)\b.{0,40}\b(function|"
                            r"method|class|program|script|code)\b", re.IGNORECASE)
_SUMMARY_WORDS = re.compile(r"\b(summari[sz]e|summary|condense|tl;?dr|"
                            r"in (?:one|1|a single) sentence|shorten)\b", re.IGNORECASE)
_NER_WORDS = re.compile(r"\b(named entit|entities|extract .{0,40}\b(person|people|"
                        r"organi[sz]ation|location|date)s?\b|label .{0,20}entit)",
                        re.IGNORECASE)
_SENTIMENT_WORDS = re.compile(r"\b(sentiment|positive or negative|"
                              r"classify .{0,40}(review|tone|feedback))\b", re.IGNORECASE)
_MC_PATTERN = re.compile(r"\bA[\).]\s.+\bB[\).]\s", re.DOTALL)
_MATH_WORDS = re.compile(r"\b(?:calculate|compute|how (?:many|much)|sum of|"
                         r"average|divided by|times|plus|minus|square root)\b|"
                         r"what is \d|%|\d+\s*[-+*/^]\s*\d+", re.IGNORECASE)
_LOGIC_WORDS = re.compile(r"\b(puzzle|riddle|if all|who is (?:the )?(?:taller|"
                          r"shorter|older|younger|shortest|tallest)|constraint|"
                          r"deduce|conclude|next (?:number|item) in)\b", re.IGNORECASE)


def classify(prompt: str) -> str:
    p = prompt or ""
    has_code = bool(_CODE_MARKERS.search(p))
    if has_code and _DEBUG_WORDS.search(p):
        return "code_debug"
    if _CODEGEN_WORDS.search(p) or (has_code and _CODEGEN_WORDS.search(p)):
        return "code_gen"
    if _SUMMARY_WORDS.search(p):
        return "summarization"
    if _NER_WORDS.search(p):
        return "ner"
    if _SENTIMENT_WORDS.search(p):
        return "sentiment"
    if _MC_PATTERN.search(p):
        return "mc"
    if _LOGIC_WORDS.search(p):
        return "logic"
    if _MATH_WORDS.search(p):
        return "math"
    return "short_qa"


@dataclass(frozen=True)
class TypeProfile:
    freeform: bool          # True: answer is the whole response, agreement off
    samples: int            # self-consistency k (local samples)
    local_max_tokens: int
    remote_max_tokens: int  # remote OUTPUT tokens are the score — keep tight
    local_style: str        # appended to the local system prompt
    remote_style: str       # the entire remote instruction — compact!


_SHORT = ("Solve the task. Think briefly if needed, then give only the final "
          "result on the last line in the form:\nAnswer: <final answer>")
_PLAIN = (" Plain text only, no markdown or LaTeX. Last line exactly:\n"
          "Answer: <final answer>")

PROFILES: dict[str, TypeProfile] = {
    "math": TypeProfile(False, 2, 256, 96, _SHORT,
                        "Answer with the final number/result only." + _PLAIN),
    "mc": TypeProfile(False, 3, 128, 16, _SHORT,
                      "Answer with the correct option letter only."),
    "sentiment": TypeProfile(False, 3, 96, 48, _SHORT,
                             "Give the sentiment label and one short justification."),
    "logic": TypeProfile(False, 2, 320, 96, _SHORT,
                         "Answer concisely with the final result." + _PLAIN),
    "short_qa": TypeProfile(False, 3, 192, 96, _SHORT,
                            "Answer accurately and concisely." + _PLAIN),
    "summarization": TypeProfile(True, 1, 260, 220,
                                 "Follow the requested format and length exactly. "
                                 "Output only the summary.",
                                 "Follow the requested format/length exactly. "
                                 "Output only the summary."),
    "ner": TypeProfile(True, 2, 220, 160,
                       "List each entity with its label. Output only the list.",
                       "List each entity with its label, nothing else."),
    "code_debug": TypeProfile(True, 1, 600, 600,
                              "Return the complete corrected code in a fenced "
                              "code block. No commentary.",
                              "Return only the complete corrected code."),
    "code_gen": TypeProfile(True, 1, 600, 600,
                            "Return the complete working code in a fenced code "
                            "block. No commentary.",
                            "Return only the complete code."),
}


def profile_for(task_type: str) -> TypeProfile:
    return PROFILES.get(task_type, PROFILES["short_qa"])


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)


def extract_answer(task_type: str, raw_text: str, extract_final_fn) -> str:
    """Type-aware final answer: fenced code for code tasks, full text for
    free-form, 'Answer: X' extraction for short-form."""
    text = (raw_text or "").strip()
    prof = profile_for(task_type)
    if task_type in ("code_debug", "code_gen"):
        blocks = _FENCE_RE.findall(text)
        if not blocks:
            return text
        # Models sometimes add a second fenced block with a usage example
        # after the real implementation; prefer the last block that actually
        # defines something over a trailing print(...)-only demo snippet.
        defs = [b for b in blocks if re.search(r"\b(def|class)\s+\w", b)]
        return (defs[-1] if defs else blocks[-1]).strip()
    if prof.freeform:
        return text
    return extract_final_fn(text)
