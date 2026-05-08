from typing import Dict, List

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

from utils_parse import split_reasoning_steps


def _safe_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute_ct_alignment(candidate_text: str, prototypes: List[Dict]) -> Dict[str, float]:
    """
    一个轻量的 CT-inspired alignment：
    - coverage: candidate steps 覆盖了多少不同 prototype
    - collapse_penalty: 是否过度集中到少数 prototype
    - ct_cost: candidate steps 与 prototype 集合整体是否对齐

    这里不是严格的 OT / CT 数学实现，而是一个稳定、可运行的近似版本。
    """
    if not candidate_text or not prototypes:
        return {
            "coverage": 0.0,
            "collapse_penalty": 0.0,
            "ct_cost": 0.0,
        }

    steps = split_reasoning_steps(candidate_text)
    if not steps:
        return {
            "coverage": 0.0,
            "collapse_penalty": 0.0,
            "ct_cost": 0.0,
        }

    proto_texts = []
    for p in prototypes:
        if not isinstance(p, dict):
            continue
        strategy_name = _safe_text(p.get("strategy_name", ""))
        description = _safe_text(p.get("description", ""))
        common_steps_raw = p.get("common_steps", []) or []
        if not isinstance(common_steps_raw, list):
            common_steps_raw = [common_steps_raw]
        common_steps = " ".join([_safe_text(s) for s in common_steps_raw])
        proto_text = f"{strategy_name} {description} {common_steps}".strip()
        proto_texts.append(proto_text if proto_text else "general reasoning")
    if not proto_texts:
        return {
            "coverage": 0.0,
            "collapse_penalty": 0.0,
            "ct_cost": 0.0,
        }

    corpus = steps + proto_texts

    if TfidfVectorizer is None or cosine_similarity is None:
        return {
            "coverage": 0.0,
            "collapse_penalty": 0.0,
            "ct_cost": 0.0,
        }

    # 每次局部 fit，一个简化但稳定的近似
    vec = TfidfVectorizer(
        max_features=256,
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b\w+\b",
    )
    try:
        X = vec.fit_transform(corpus)
    except ValueError:
        return {
            "coverage": 0.0,
            "collapse_penalty": 0.0,
            "ct_cost": 0.0,
        }

    step_x = X[:len(steps)]
    proto_x = X[len(steps):]

    sim = cosine_similarity(step_x, proto_x)
    if sim.size == 0 or sim.shape[1] == 0:
        return {
            "coverage": 0.0,
            "collapse_penalty": 0.0,
            "ct_cost": 0.0,
        }

    # 每个 reasoning step 最接近哪个 prototype
    step_best = sim.max(axis=1)              # shape: [n_steps]
    best_proto_idx = sim.argmax(axis=1)      # shape: [n_steps]

    # 1) coverage：用了多少不同 prototype
    unique_used = len(set(best_proto_idx.tolist())) if len(best_proto_idx) else 0
    coverage = unique_used / max(1, len(prototypes))
    coverage = _clamp01(coverage)

    # 2) collapse penalty：
    # 如果所有步骤都压到一个 prototype，上升；如果分散得更均匀，下降
    usage_counts = np.bincount(best_proto_idx, minlength=len(prototypes)).astype(np.float32)
    usage_probs = usage_counts / max(1.0, usage_counts.sum())

    # 归一化熵，越高说明越分散，越低说明越塌
    nonzero = usage_probs[usage_probs > 0]
    entropy = -np.sum(nonzero * np.log(nonzero + 1e-12))
    max_entropy = np.log(len(prototypes)) if len(prototypes) > 1 else 1.0
    normalized_entropy = float(entropy / max(max_entropy, 1e-12))
    normalized_entropy = _clamp01(normalized_entropy)

    collapse_penalty = 1.0 - normalized_entropy
    collapse_penalty = _clamp01(collapse_penalty)

    # 3) ct cost：
    # candidate steps 和 prototype 的整体对齐程度越高，cost 越低
    mean_alignment = float(np.mean(step_best)) if len(step_best) else 0.0
    mean_alignment = _clamp01(mean_alignment)
    ct_cost = 1.0 - mean_alignment
    ct_cost = _clamp01(ct_cost)

    return {
        "coverage": float(coverage),
        "collapse_penalty": float(collapse_penalty),
        "ct_cost": float(ct_cost),
    }
