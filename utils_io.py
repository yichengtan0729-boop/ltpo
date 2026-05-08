import json
import os
from typing import Any, Dict, Iterable, List


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def safe_name_from_path(name_or_path: str) -> str:
    text = str(name_or_path or "").replace("\\", "/").rstrip("/")
    name = text.rsplit("/", 1)[-1] if text else ""
    return name.replace(":", "_")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return items

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(path: str, obj: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
