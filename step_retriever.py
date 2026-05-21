import os
import pickle
import re
from typing import Any, Dict, List

import numpy as np

from utils_io import load_json


class StepPrototypeRetriever:
    def __init__(self, prototype_path: str, vectorizer_path: str = ""):
        self.prototype_path = prototype_path
        self.prototypes: List[Dict[str, Any]] = []
        self.vectorizer = None

        if not prototype_path or not os.path.exists(prototype_path):
            raise FileNotFoundError(f"Step prototype file not found: {prototype_path}")

        data = load_json(prototype_path)
        if isinstance(data, dict):
            raw = data.get("prototypes", [])
            json_vectorizer_path = data.get("vectorizer_path", "")
        elif isinstance(data, list):
            raw = data
            json_vectorizer_path = ""
        else:
            raw = []
            json_vectorizer_path = ""

        self.prototypes = [p for p in raw if isinstance(p, dict)]
        self.vectorizer_path = self._resolve_vectorizer_path(vectorizer_path or json_vectorizer_path)
        if self.vectorizer_path and os.path.exists(self.vectorizer_path):
            try:
                with open(self.vectorizer_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
            except Exception:
                self.vectorizer = None

    def _resolve_vectorizer_path(self, path: str) -> str:
        if not path:
            return ""
        if os.path.exists(path):
            return path
        if not os.path.isabs(path):
            candidate = os.path.join(os.path.dirname(self.prototype_path), path)
            if os.path.exists(candidate):
                return candidate
        return path

    def _safe_float(self, x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def _query_vector(self, question: str):
        if self.vectorizer is None:
            return None
        try:
            return self.vectorizer.transform([question or ""]).toarray().astype(np.float32)[0]
        except Exception:
            return None

    def _cosine(self, qv, centroid) -> float:
        try:
            q = np.asarray(qv, dtype=np.float32).reshape(-1)
            c = np.asarray(centroid, dtype=np.float32).reshape(-1)
            if q.shape[0] != c.shape[0] or q.shape[0] == 0:
                return 0.0
            denom = max(float(np.linalg.norm(q) * np.linalg.norm(c)), 1e-12)
            return float(np.dot(q, c) / denom)
        except Exception:
            return 0.0

    def _lexical_overlap(self, question: str, proto: Dict[str, Any]) -> float:
        query_terms = set(re.findall(r"[a-z0-9]+", (question or "").lower()))
        if not query_terms:
            return 0.0
        pieces = [
            proto.get("scenario", ""),
            proto.get("role", ""),
            proto.get("step_type", ""),
            proto.get("prototype_text", ""),
            " ".join(str(x) for x in proto.get("common_steps", []) or []),
        ]
        proto_terms = set(re.findall(r"[a-z0-9]+", " ".join(str(x) for x in pieces).lower()))
        return float(len(query_terms & proto_terms) / max(1, len(query_terms)))

    def _similarity(self, question: str, proto: Dict[str, Any], qv=None) -> float:
        if qv is not None and proto.get("centroid") is not None:
            sim = self._cosine(qv, proto.get("centroid"))
            if sim != 0.0:
                return sim
        return self._lexical_overlap(question, proto)

    def _score(
        self,
        question: str,
        proto: Dict[str, Any],
        scenario: str,
        role: str,
        qv=None,
        alpha: float = 0.60,
        beta: float = 0.25,
        gamma: float = 0.10,
        delta: float = 0.05,
        eta: float = 0.10,
    ) -> float:
        sim = self._similarity(question, proto, qv=qv)
        reliability = self._safe_float(proto.get("reliability", proto.get("reliability_mean", 0.0)))
        role_match = 1.0 if proto.get("role") == role else 0.0
        scenario_match = 1.0 if proto.get("scenario") == scenario else 0.0
        failure_risk = self._safe_float(proto.get("failure_risk", 0.0))
        return float(
            alpha * sim
            + beta * reliability
            + gamma * role_match
            + delta * scenario_match
            - eta * failure_risk
        )

    def _normalize(self, proto: Dict[str, Any], score: float) -> Dict[str, Any]:
        out = dict(proto)
        out["retrieval_score"] = float(score)
        out.setdefault("common_steps", [])
        out.setdefault("prototype_text", "")
        out.setdefault("reliability", self._safe_float(out.get("reliability_mean", 0.0)))
        out.setdefault("failure_risk", self._safe_float(out.get("failure_risk", 0.0)))
        return out

    def retrieve_step_plan(
        self,
        question: str,
        scenario: str,
        roles: List[str],
        top_k_per_role: int = 1,
        memory_type: str = "verified_solution",
    ) -> List[Dict]:
        try:
            if not self.prototypes or not roles:
                return []

            qv = self._query_vector(question)
            results: List[Dict] = []
            limit = max(1, int(top_k_per_role or 1))

            for role in roles:
                pools = [
                    [
                        p for p in self.prototypes
                        if p.get("memory_type") == memory_type
                        and p.get("role") == role
                        and p.get("scenario") == scenario
                    ],
                    [
                        p for p in self.prototypes
                        if p.get("memory_type") == memory_type
                        and p.get("role") == role
                        and p.get("scenario") == "general_reasoning"
                    ],
                    [
                        p for p in self.prototypes
                        if p.get("memory_type") == memory_type
                        and p.get("role") == role
                    ],
                ]

                candidates: List[Dict] = []
                for pool in pools:
                    if pool:
                        candidates = pool
                        break
                if not candidates:
                    continue

                scored = [
                    (self._score(question, p, scenario=scenario, role=role, qv=qv), p)
                    for p in candidates
                ]
                scored.sort(key=lambda x: x[0], reverse=True)
                results.extend(self._normalize(p, score) for score, p in scored[:limit])

            return results
        except Exception:
            return []

    def retrieve_failure_prototypes(
        self,
        question: str,
        scenario: str,
        top_k: int = 4,
    ) -> List[Dict]:
        try:
            limit = max(0, int(top_k or 0))
            if limit == 0 or not self.prototypes:
                return []

            qv = self._query_vector(question)
            failures = [p for p in self.prototypes if p.get("memory_type") == "failure"]
            if not failures:
                return []

            scored = []
            for proto in failures:
                sim = self._similarity(question, proto, qv=qv)
                risk = self._safe_float(proto.get("failure_risk", proto.get("reliability", 0.0)))
                scenario_bonus = 0.05 if proto.get("scenario") == scenario else 0.0
                scored.append((float(sim * max(risk, 0.0) + scenario_bonus), proto))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [self._normalize(p, score) for score, p in scored[:limit]]
        except Exception:
            return []
