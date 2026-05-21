import math
import re
from typing import Dict, List


DEFAULT_STEP_ROLES = [
    "understand_extract",
    "plan_retrieve",
    "infer_compute",
    "verify_correct",
]


_STEP_MARKER_RE = re.compile(
    r"(?i)(?:^|\s)(step\s*\d+|first|second|third|fourth|next|then|therefore|so|finally)\s*[:.,-]?\s+"
)


def _clean_step_text(text: str) -> str:
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", text or "")
    text = re.sub(r"(?i)^\s*step\s*\d+\s*[:.)-]?\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_rationale(rationale: str) -> List[str]:
    text = (rationale or "").strip()
    if not text:
        return []

    # Put explicit discourse markers on their own boundary before sentence splitting.
    text = re.sub(r"(?i)(step\s*\d+\s*[:.)-])", r"\n\1 ", text)
    text = re.sub(
        r"(?i)\b(first|second|third|fourth|next|then|therefore|so|finally)\b\s*[,.:;-]?",
        r"\n\1 ",
        text,
    )

    chunks: List[str] = []
    for block in re.split(r"\n+", text):
        block = block.strip()
        if not block:
            continue
        parts = re.split(r"(?<=[.!?])\s+", block)
        for part in parts:
            cleaned = _clean_step_text(part)
            if cleaned:
                chunks.append(cleaned)

    # If the rationale is a single long paragraph with little punctuation, use marker splits.
    if len(chunks) <= 1:
        marked = _STEP_MARKER_RE.split(text)
        chunks = [_clean_step_text(x) for x in marked if _clean_step_text(x)]

    return chunks


def _merge_to_max_steps(chunks: List[str], max_steps: int) -> List[str]:
    if max_steps <= 0:
        return []
    if len(chunks) <= max_steps:
        return chunks

    merged: List[str] = []
    bucket_size = int(math.ceil(len(chunks) / float(max_steps)))
    for start in range(0, len(chunks), bucket_size):
        merged.append(" ".join(chunks[start:start + bucket_size]).strip())
        if len(merged) == max_steps - 1:
            tail = " ".join(chunks[start + bucket_size:]).strip()
            if tail:
                merged.append(tail)
            break
    return [x for x in merged if x][:max_steps]


def infer_step_type(step_text: str) -> str:
    text = (step_text or "").lower()
    rules = [
        (("condition", "extract", "given", "known"), "condition_extraction"),
        (("equation", "variable", "let", "solve"), "relation_setup"),
        (("calculate", "compute", "evaluate"), "computation"),
        (("check", "verify", "final", "therefore"), "verification"),
        (("wrong", "mistake", "correct", "revise"), "correction"),
    ]
    for keywords, step_type in rules:
        if any(key in text for key in keywords):
            return step_type
    return "general_reasoning"


def parse_steps(
    question: str,
    rationale: str,
    dataset_name: str = "",
    max_steps: int = 4,
) -> List[Dict]:
    try:
        if not rationale or not str(rationale).strip():
            return []

        limit = max(0, int(max_steps or 0))
        if limit <= 0:
            return []

        chunks = _split_rationale(str(rationale))
        chunks = [_clean_step_text(x) for x in chunks if _clean_step_text(x)]
        chunks = _merge_to_max_steps(chunks, limit)

        parsed: List[Dict] = []
        for idx, step_text in enumerate(chunks):
            role = DEFAULT_STEP_ROLES[min(idx, len(DEFAULT_STEP_ROLES) - 1)]
            parsed.append(
                {
                    "step_id": idx,
                    "role": role,
                    "step_type": infer_step_type(step_text),
                    "step_text": step_text,
                }
            )
        return parsed
    except Exception:
        return []
