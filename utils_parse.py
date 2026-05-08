import json
import re
from typing import List, Optional


_BOXED_INLINE_RE = re.compile(r"\\boxed\s+([^\s]+)")
_FINAL_RE = re.compile(r"(?:final answer\s*[:：]\s*)(.+)$", re.IGNORECASE | re.MULTILINE)
_ANS_RE = re.compile(r"(?:answer\s*[:：]\s*)(.+)$", re.IGNORECASE | re.MULTILINE)
_HASH_ANS_RE = re.compile(r"####\s*(.+)$", re.MULTILINE)
_JSON_FINAL_RE = re.compile(r'"(?:final answer|final_answer|answer)"\s*:\s*"([^"]+)"', re.IGNORECASE)


def extract_boxed_answer(text: str) -> Optional[str]:
    text = text or ""
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start >= 0:
        i = start + len(marker)
        depth = 1
        chars = []
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(chars).strip()
            chars.append(ch)
            i += 1

    matches = _BOXED_INLINE_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return None


def extract_json_final_answer(text: str) -> Optional[str]:
    if not text:
        return None

    m = _JSON_FINAL_RE.search(text)
    if m:
        return m.group(1).strip()

    # 尝试整体 json 解析
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                for key in ("final answer", "final_answer", "answer"):
                    if key in obj:
                        return str(obj[key]).strip()
        except Exception:
            pass
    return None


def extract_final_answer_fallback(text: str) -> Optional[str]:
    if not text:
        return None

    json_ans = extract_json_final_answer(text)
    if json_ans:
        return json_ans

    m = _HASH_ANS_RE.search(text)
    if m:
        return m.group(1).strip()

    for pattern in (_FINAL_RE, _ANS_RE):
        m = pattern.search(text)
        if m:
            return m.group(1).strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[-1][:200].strip()


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None

    ans = str(answer).strip()
    ans = ans.replace("$", "")
    ans = ans.replace(",", "")
    ans = ans.replace("。", ".")
    ans = re.sub(r"\s+", " ", ans).strip()
    ans = re.sub(r"^\\text\{(.+)\}$", r"\1", ans).strip()
    ans = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s+is\s+", "", ans, flags=re.IGNORECASE).strip()
    if len(ans) >= 2 and ans[0] == "{" and ans[-1] == "}":
        ans = ans[1:-1].strip()

    # 去掉末尾多余句号
    if ans.endswith(".") and len(ans) > 1:
        ans = ans[:-1].strip()

    return ans if ans else None


def safe_parse_answer(text: str) -> Optional[str]:
    ans = extract_boxed_answer(text)
    if ans:
        return normalize_answer(ans)

    ans = extract_final_answer_fallback(text)
    return normalize_answer(ans) if ans else None


def split_reasoning_steps(text: str) -> List[str]:
    if not text:
        return []

    chunks: List[str] = []
    for raw in re.split(r"\n+|(?:(?<=\.)\s+)", text):
        raw = raw.strip()
        if not raw:
            continue
        chunks.append(raw)
    return chunks
