import os
import pickle
from typing import Any, Dict, List

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception as exc:
    KMeans = None
    TfidfVectorizer = None
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None

from utils_io import read_jsonl, save_json, ensure_dir, safe_name_from_path
from utils_parse import split_reasoning_steps


def _top_indices_by_value(vals: List[float], k: int) -> List[int]:
    if not vals:
        return []
    arr = np.asarray(vals, dtype=np.float32)
    return list(np.argsort(arr)[::-1][:k])


def _summarize_strategy(texts: List[str]) -> str:
    if not texts:
        return "general_reasoning"

    joined = " ".join(texts[:5]).lower()

    keywords = [
        ("equation", "equation_modeling"),
        ("variable", "equation_modeling"),
        ("solve for", "equation_modeling"),
        ("case", "case_analysis"),
        ("cases", "case_analysis"),
        ("enumerate", "enumeration_verification"),
        ("count", "counting_enumeration"),
        ("verify", "verification_check"),
        ("check", "verification_check"),
        ("constraint", "constraint_check"),
        ("date", "date_reasoning"),
        ("calendar", "date_reasoning"),
        ("yes", "binary_reasoning"),
        ("no", "binary_reasoning"),
        ("probability", "probability_reasoning"),
    ]
    for key, name in keywords:
        if key in joined:
            return name
    return "general_reasoning"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _short_text(text: str, max_chars: int = 160) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip() + "..."


def _strategy_description(strategy_name: str) -> str:
    descriptions = {
        "equation_modeling": "Set up the key variables, solve the equations, and verify the result.",
        "case_analysis": "Split the problem into a few cases and eliminate inconsistent cases.",
        "enumeration_verification": "Enumerate the valid possibilities and check them against the constraints.",
        "counting_enumeration": "Count the relevant cases carefully and verify no case is double-counted.",
        "verification_check": "Reason step by step, then check the final answer against the question.",
        "constraint_check": "Track the constraints explicitly and discard options that violate them.",
        "date_reasoning": "Translate the calendar information into ordered date or weekday steps.",
        "binary_reasoning": "Evaluate the claim directly and answer with the supported yes/no result.",
        "probability_reasoning": "Identify the sample space and compute the favorable outcomes.",
    }
    return descriptions.get(strategy_name, "Break the problem into short steps and verify the final answer.")


def _output_paths(args) -> Dict[str, str]:
    prototype_dir = args.prototype_dir or os.path.join(args.output_dir, "prototypes")
    ensure_dir(prototype_dir)

    data_name = safe_name_from_path(getattr(args, "memory_dataset", None) or args.dataset)
    model_name = safe_name_from_path(args.model_name_or_path)
    prototype_json_path = (
        args.prototype_path
        if getattr(args, "prototype_path", None)
        else os.path.join(prototype_dir, f"{model_name}-{data_name}-prototypes.json")
    )
    vectorizer_path = os.path.splitext(prototype_json_path)[0] + ".vectorizer.pkl"
    return {
        "prototype_path": prototype_json_path,
        "vectorizer_path": vectorizer_path,
    }


def _build_memory_path(args) -> str:
    if getattr(args, "memory_output_path", None):
        return args.memory_output_path

    data_name = safe_name_from_path(getattr(args, "memory_dataset", None) or args.dataset)
    model_name = safe_name_from_path(args.model_name_or_path)
    memory_dir = args.memory_dir or os.path.join(args.output_dir, "memories")
    return os.path.join(memory_dir, f"{model_name}-{data_name}-memory.jsonl")


def build_prototypes(args) -> Dict[str, str]:
    memory_path = _build_memory_path(args)
    paths = _output_paths(args)
    prototype_json_path = paths["prototype_path"]
    vectorizer_path = paths["vectorizer_path"]

    if not os.path.exists(memory_path):
        raise FileNotFoundError(f"No memory file found at {memory_path}")

    items = read_jsonl(memory_path)
    if not items:
        save_json(
            prototype_json_path,
            {
                "prototypes": [],
                "vectorizer_path": "",
                "memory_path": memory_path,
            },
        )
        return paths

    if KMeans is None or TfidfVectorizer is None:
        raise ImportError(
            "build_prototypes requires scikit-learn. The current environment could not import it."
        ) from _SKLEARN_IMPORT_ERROR

    texts = [
        it.get("embedding_text")
        or it.get("chosen_rationale")
        or it.get("question", "")
        for it in items
    ]
    texts = [text if str(text).strip() else "general reasoning" for text in texts]

    vectorizer = TfidfVectorizer(
        max_features=getattr(args, "prototype_max_features", 4096),
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b\w+\b",
    )
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        save_json(
            prototype_json_path,
            {
                "prototypes": [],
                "vectorizer_path": "",
                "memory_path": memory_path,
            },
        )
        return paths

    n_clusters = min(max(1, getattr(args, "n_prototypes", 16)), len(items))
    kmeans = KMeans(n_clusters=n_clusters, random_state=getattr(args, "seed", 42), n_init=10)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    prototypes: List[Dict[str, Any]] = []

    for c in range(n_clusters):
        member_indices = [i for i, lab in enumerate(labels) if lab == c]
        member_items = [items[i] for i in member_indices]

        member_scores = [_safe_float(it.get("reliability", 0.0)) for it in member_items]
        rep_local = _top_indices_by_value(
            member_scores,
            min(getattr(args, "top_examples_per_prototype", 3), len(member_items)),
        )
        rep_items = [member_items[j] for j in rep_local]

        rep_texts = [it.get("chosen_rationale", "") for it in rep_items]
        strategy_name = _summarize_strategy(rep_texts)

        common_steps: List[str] = []
        for txt in rep_texts[:3]:
            common_steps.extend(split_reasoning_steps(txt)[:2])

        # 去重保序
        dedup_steps = []
        seen = set()
        for step in common_steps:
            step_norm = step.strip().lower()
            if not step_norm or step_norm in seen:
                continue
            seen.add(step_norm)
            dedup_steps.append(_short_text(step, max_chars=160))
        common_steps = dedup_steps[:3]

        description = _strategy_description(strategy_name)

        prototype = {
            "prototype_id": f"p_{c:03d}",
            "strategy_name": strategy_name,
            "description": description,
            "common_steps": common_steps,
            "member_ids": [it.get("id", f"memory_{member_indices[j]}") for j, it in enumerate(member_items)],
            "representative_examples": [
                {
                    "id": it.get("id", ""),
                    "question": it.get("question", ""),
                    "chosen_answer": it.get("chosen_answer"),
                    "reliability": _safe_float(it.get("reliability", 0.0)),
                }
                for it in rep_items
            ],
            "reliability_mean": float(np.mean(member_scores)) if member_scores else 0.0,
            "reliability": float(np.mean(member_scores)) if member_scores else 0.0,
            "centroid": centroids[c].tolist(),
            "support_size": len(member_items),
        }
        prototypes.append(prototype)

    save_json(
        prototype_json_path,
        {
            "prototypes": prototypes,
            "vectorizer_path": vectorizer_path,
            "memory_path": memory_path,
        },
    )

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    return {
        "prototype_path": prototype_json_path,
        "vectorizer_path": vectorizer_path,
    }
