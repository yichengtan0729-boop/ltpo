import os
import pickle
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

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

from utils_io import ensure_dir, read_jsonl, safe_name_from_path, save_json


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _short_text(text: str, max_chars: int = 200) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def _step_memory_path(args) -> str:
    if getattr(args, "step_memory_output_path", None):
        return args.step_memory_output_path
    data_name = safe_name_from_path(getattr(args, "memory_dataset", None) or args.dataset)
    model_name = safe_name_from_path(args.model_name_or_path)
    memory_dir = args.step_memory_dir or os.path.join(args.output_dir, "step_memories")
    return os.path.join(memory_dir, f"{model_name}-{data_name}-step-memory.jsonl")


def _output_paths(args) -> Dict[str, str]:
    prototype_dir = args.step_prototype_dir or os.path.join(args.output_dir, "step_prototypes")
    ensure_dir(prototype_dir)
    data_name = safe_name_from_path(getattr(args, "memory_dataset", None) or args.dataset)
    model_name = safe_name_from_path(args.model_name_or_path)
    prototype_path = (
        args.step_prototype_path
        if getattr(args, "step_prototype_path", None)
        else os.path.join(prototype_dir, f"{model_name}-{data_name}-step-prototypes.json")
    )
    vectorizer_path = os.path.splitext(prototype_path)[0] + ".step.vectorizer.pkl"
    return {
        "prototype_path": prototype_path,
        "vectorizer_path": vectorizer_path,
    }


def _matrix_mean_row(X) -> List[float]:
    arr = np.asarray(X.mean(axis=0)).reshape(-1).astype(np.float32)
    return arr.tolist()


def _top_indices_by_reliability(items: List[Dict[str, Any]], k: int) -> List[int]:
    scores = np.asarray([_safe_float(it.get("step_reliability", 0.0)) for it in items], dtype=np.float32)
    if scores.size == 0:
        return []
    return list(np.argsort(scores)[::-1][:k])


def _dedup_steps(items: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        step = _short_text(item.get("step_text", ""), max_chars=180)
        key = step.lower()
        if not step or key in seen:
            continue
        seen.add(key)
        out.append(step)
        if len(out) >= limit:
            break
    return out


def _prototype_id(
    scenario: str,
    role: str,
    memory_type: str,
    cluster_idx: int,
) -> str:
    suffix = "verified" if memory_type == "verified_solution" else memory_type
    safe_parts = [safe_name_from_path(x).lower() for x in (scenario, role, suffix)]
    return f"{safe_parts[0]}_{safe_parts[1]}_{safe_parts[2]}_p{cluster_idx:03d}"


def _build_prototype(
    group_key: Tuple[str, str, str],
    member_items: List[Dict[str, Any]],
    member_indices: List[int],
    centroid: List[float],
    cluster_idx: int,
    top_examples: int,
) -> Dict[str, Any]:
    scenario, role, memory_type = group_key
    rep_local = _top_indices_by_reliability(member_items, min(top_examples, len(member_items)))
    rep_items = [member_items[i] for i in rep_local]
    if not rep_items:
        rep_items = member_items[:1]

    reliability_scores = [_safe_float(it.get("step_reliability", 0.0)) for it in member_items]
    reliability_mean = float(np.mean(reliability_scores)) if reliability_scores else 0.0
    step_types = [str(it.get("step_type", "general_reasoning")) for it in member_items]
    step_type = Counter(step_types).most_common(1)[0][0] if step_types else "general_reasoning"
    common_steps = _dedup_steps(rep_items, limit=3)
    prototype_text = common_steps[0] if common_steps else _short_text(rep_items[0].get("step_text", ""))

    if memory_type == "failure":
        failure_risk = reliability_mean
    elif memory_type == "correction":
        failure_risk = 0.2
    else:
        failure_risk = 0.0

    return {
        "prototype_id": _prototype_id(scenario, role, memory_type, cluster_idx),
        "scenario": scenario,
        "role": role,
        "step_type": step_type,
        "memory_type": memory_type,
        "prototype_text": prototype_text,
        "common_steps": common_steps,
        "representative_examples": [
            {
                "id": it.get("id", ""),
                "question": it.get("question", ""),
                "step_text": it.get("step_text", ""),
                "parsed_answer": it.get("parsed_answer", ""),
                "true_answer": it.get("true_answer", ""),
                "step_reliability": _safe_float(it.get("step_reliability", 0.0)),
                "memory_type": it.get("memory_type", memory_type),
            }
            for it in rep_items
        ],
        "member_ids": [it.get("id", f"step_memory_{idx}") for idx, it in zip(member_indices, member_items)],
        "centroid": centroid,
        "reliability_mean": reliability_mean,
        "reliability": reliability_mean,
        "failure_risk": float(failure_risk),
        "support_size": len(member_items),
    }


def build_step_prototypes(args) -> Dict[str, str]:
    memory_path = _step_memory_path(args)
    paths = _output_paths(args)
    prototype_path = paths["prototype_path"]
    vectorizer_path = paths["vectorizer_path"]

    if not os.path.exists(memory_path):
        raise FileNotFoundError(f"No step memory file found at {memory_path}")

    items = [it for it in read_jsonl(memory_path) if isinstance(it, dict)]
    metadata = {
        "num_verified": sum(1 for it in items if it.get("memory_type") == "verified_solution"),
        "num_failure": sum(1 for it in items if it.get("memory_type") == "failure"),
        "num_correction": sum(1 for it in items if it.get("memory_type") == "correction"),
    }

    if not items:
        save_json(
            prototype_path,
            {
                "prototypes": [],
                "vectorizer_path": "",
                "memory_path": memory_path,
                "metadata": metadata,
            },
        )
        return paths

    if KMeans is None or TfidfVectorizer is None:
        raise ImportError("build_step_prototypes requires scikit-learn.") from _SKLEARN_IMPORT_ERROR

    texts = [
        it.get("embedding_text")
        or " ".join(
            str(x)
            for x in [
                it.get("scenario", ""),
                it.get("role", ""),
                it.get("step_type", ""),
                it.get("question", ""),
                it.get("step_text", ""),
            ]
        )
        for it in items
    ]
    texts = [str(t).strip() if str(t).strip() else "general reasoning" for t in texts]

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
            prototype_path,
            {
                "prototypes": [],
                "vectorizer_path": "",
                "memory_path": memory_path,
                "metadata": metadata,
            },
        )
        return paths

    groups: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        key = (
            str(item.get("scenario", "general_reasoning")),
            str(item.get("role", "infer_compute")),
            str(item.get("memory_type", "verified_solution")),
        )
        groups[key].append(idx)

    prototypes: List[Dict[str, Any]] = []
    per_group = max(1, int(getattr(args, "n_step_prototypes_per_group", 4)))
    top_examples = max(1, int(getattr(args, "top_examples_per_prototype", 3)))
    random_state = int(getattr(args, "seed", 42))

    for group_key, indices in sorted(groups.items(), key=lambda x: x[0]):
        group_items = [items[i] for i in indices]
        if len(indices) <= 2 or len(indices) <= per_group:
            centroid = _matrix_mean_row(X[indices])
            prototypes.append(
                _build_prototype(
                    group_key=group_key,
                    member_items=group_items,
                    member_indices=indices,
                    centroid=centroid,
                    cluster_idx=0,
                    top_examples=top_examples,
                )
            )
            continue

        n_clusters = min(per_group, len(indices))
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X[indices])
            centroids = kmeans.cluster_centers_
        except Exception:
            centroid = _matrix_mean_row(X[indices])
            prototypes.append(
                _build_prototype(
                    group_key=group_key,
                    member_items=group_items,
                    member_indices=indices,
                    centroid=centroid,
                    cluster_idx=0,
                    top_examples=top_examples,
                )
            )
            continue

        for cluster_idx in range(n_clusters):
            local_member_positions = [j for j, label in enumerate(labels) if int(label) == cluster_idx]
            if not local_member_positions:
                continue
            member_indices = [indices[j] for j in local_member_positions]
            member_items = [items[idx] for idx in member_indices]
            prototypes.append(
                _build_prototype(
                    group_key=group_key,
                    member_items=member_items,
                    member_indices=member_indices,
                    centroid=np.asarray(centroids[cluster_idx], dtype=np.float32).reshape(-1).tolist(),
                    cluster_idx=cluster_idx,
                    top_examples=top_examples,
                )
            )

    save_json(
        prototype_path,
        {
            "prototypes": prototypes,
            "vectorizer_path": vectorizer_path,
            "memory_path": memory_path,
            "metadata": metadata,
        },
    )

    ensure_dir(os.path.dirname(vectorizer_path))
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    return {
        "prototype_path": prototype_path,
        "vectorizer_path": vectorizer_path,
    }
