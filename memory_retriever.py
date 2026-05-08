import os
import pickle
import re
from typing import Any, Dict, List

import numpy as np

from utils_io import load_json


class PrototypeRetriever:
    def __init__(self, prototype_path: str, vectorizer_path: str = ""):
        if not os.path.exists(prototype_path):
            raise FileNotFoundError(f"Prototype file not found: {prototype_path}")

        data = load_json(prototype_path)
        if isinstance(data, dict):
            raw_prototypes = data.get("prototypes", [])
            if not isinstance(raw_prototypes, list) and "prototype_id" in data:
                raw_prototypes = [data]
            json_vectorizer_path = data.get("vectorizer_path", "")
        elif isinstance(data, list):
            raw_prototypes = data
            json_vectorizer_path = ""
        else:
            raw_prototypes = []
            json_vectorizer_path = ""

        self.prototypes = [p for p in raw_prototypes if isinstance(p, dict)]
        self.vectorizer = None
        self.centroid_indices: List[int] = []
        self.centroids = np.zeros((0, 0), dtype=np.float32)

        self.vectorizer_path = self._resolve_vectorizer_path(
            vectorizer_path if vectorizer_path else json_vectorizer_path,
            prototype_path,
        )
        if self.vectorizer_path and os.path.exists(self.vectorizer_path):
            try:
                with open(self.vectorizer_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
            except Exception:
                self.vectorizer = None

        self.centroids = self._build_centroids(self.prototypes)

    def _resolve_vectorizer_path(self, vectorizer_path: str, prototype_path: str) -> str:
        if not vectorizer_path:
            return ""
        if os.path.exists(vectorizer_path):
            return vectorizer_path
        if not os.path.isabs(vectorizer_path):
            candidate = os.path.join(os.path.dirname(prototype_path), vectorizer_path)
            if os.path.exists(candidate):
                return candidate
        return vectorizer_path

    def _safe_float(self, x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def _cosine_similarity(self, qv: np.ndarray, mat: np.ndarray) -> np.ndarray:
        q = np.asarray(qv, dtype=np.float32).reshape(-1)
        m = np.asarray(mat, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        m_norm = np.linalg.norm(m, axis=1)
        denom = np.maximum(q_norm * m_norm, 1e-12)
        return (m @ q) / denom

    def _build_centroids(self, prototypes: List[Dict]) -> np.ndarray:
        centroids = []
        valid_indices = []

        for idx, p in enumerate(prototypes):
            centroid = p.get("centroid", None)
            if centroid is None:
                continue
            arr = np.asarray(centroid, dtype=np.float32)
            if arr.ndim != 1:
                continue
            centroids.append(arr)
            valid_indices.append(idx)

        if not centroids:
            self.centroid_indices = []
            return np.zeros((0, 0), dtype=np.float32)

        dim_counts: Dict[int, int] = {}
        for arr in centroids:
            dim_counts[int(arr.shape[0])] = dim_counts.get(int(arr.shape[0]), 0) + 1
        keep_dim = max(dim_counts, key=dim_counts.get)

        filtered = [(idx, arr) for idx, arr in zip(valid_indices, centroids) if int(arr.shape[0]) == keep_dim]
        self.centroid_indices = [idx for idx, _ in filtered]
        return np.stack([arr for _, arr in filtered], axis=0)

    def _prototype_reliability(self, p: Dict) -> float:
        r_mean = self._safe_float(p.get("reliability_mean", 0.0))
        r = self._safe_float(p.get("reliability", 0.0))
        return float(max(r_mean, r, 0.05))

    def _normalize_prototype(self, p: Dict, retrieval_score: float) -> Dict:
        out = dict(p)
        out.setdefault("prototype_id", "")
        out.setdefault("strategy_name", "")
        out.setdefault("description", "")
        out.setdefault("support_size", 0)

        common_steps = out.get("common_steps", [])
        if not isinstance(common_steps, list):
            common_steps = [str(common_steps)] if common_steps else []
        out["common_steps"] = common_steps

        out["retrieval_score"] = float(retrieval_score)
        out.setdefault("reliability_mean", self._prototype_reliability(out))
        out.setdefault("reliability", self._prototype_reliability(out))
        return out

    def _score_with_centroids(self, question: str) -> List:
        if self.vectorizer is None or self.centroids.size == 0:
            return []
        try:
            qv = self.vectorizer.transform([question or ""]).toarray().astype(np.float32)
        except Exception:
            return []
        if qv.ndim != 2 or qv.shape[1] != self.centroids.shape[1]:
            return []

        sims = self._cosine_similarity(qv[0], self.centroids)
        scored = []
        for row_idx, proto_idx in enumerate(self.centroid_indices):
            p = self.prototypes[proto_idx]
            sim = self._safe_float(sims[row_idx], 0.0)
            reliability = self._prototype_reliability(p)
            scored.append((float(sim * reliability), proto_idx))
        return scored

    def _score_lexical(self, question: str) -> List:
        query_terms = set(re.findall(r"[a-z0-9]+", (question or "").lower()))
        scored = []
        for idx, p in enumerate(self.prototypes):
            common_steps = p.get("common_steps", []) or []
            if not isinstance(common_steps, list):
                common_steps = [str(common_steps)]
            proto_text = " ".join([
                str(p.get("strategy_name", "")),
                str(p.get("description", "")),
                " ".join(str(s) for s in common_steps),
            ]).lower()
            proto_terms = set(re.findall(r"[a-z0-9]+", proto_text))
            overlap = len(query_terms & proto_terms) / max(1, len(query_terms))
            reliability = self._prototype_reliability(p)
            scored.append((float(0.70 * overlap + 0.30 * reliability), idx))
        return scored

    def retrieve(self, question: str, top_k: int = 4) -> List[Dict]:
        limit = max(0, int(top_k or 0))
        if limit == 0 or not self.prototypes:
            return []

        scored = self._score_with_centroids(question)
        if not scored:
            scored = self._score_lexical(question)

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            self._normalize_prototype(self.prototypes[idx], retrieval_score=score)
            for score, idx in scored[:limit]
        ]
