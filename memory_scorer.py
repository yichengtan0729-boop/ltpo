from typing import Dict, List, Optional, Any

from ct_alignment import compute_ct_alignment
from utils_parse import safe_parse_answer


DEFAULT_WEIGHTS: Dict[str, float] = {
    "answer_consistency": 0.55,
    "memory_support": 0.20,
    "validity": 0.25,
    "prototype_coverage": 0.00,
    "confidence": 0.00,
    "copy_penalty": 0.02,
    "collapse_penalty": 0.00,
    "ct_cost": 0.00,
}


class MemoryScorer:
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        enable_ct: bool = True,
        enable_copy_penalty: bool = True,
    ):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.enable_ct = enable_ct
        self.enable_copy_penalty = enable_copy_penalty

    def _safe_float(self, x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def _clamp01(self, x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def _weight(self, key: str) -> float:
        if not self.enable_ct and key in {"prototype_coverage", "collapse_penalty", "ct_cost"}:
            return 0.0
        if not self.enable_copy_penalty and key == "copy_penalty":
            return 0.0
        return self._safe_float(self.weights.get(key, DEFAULT_WEIGHTS.get(key, 0.0)))

    def _memory_support(self, candidate_text: str, prototypes: List[Dict]) -> float:
        """
        一个轻量、可解释的 memory support：
        1. 看 common_steps / description / strategy_name 是否和 candidate 有 lexical overlap
        2. 综合 prototype 的 retrieval_score / reliability_mean / reliability
        """
        if not prototypes:
            return 0.0

        lower = (candidate_text or "").lower()
        best_score = 0.0

        for p in prototypes:
            if not isinstance(p, dict):
                continue
            token_hits = 0

            # 1) common_steps 匹配
            common_steps = p.get("common_steps", []) or []
            if not isinstance(common_steps, list):
                common_steps = [str(common_steps)]
            for step in common_steps:
                step = (step or "").strip().lower()
                if not step:
                    continue
                if step[:40] in lower:
                    token_hits += 1

            # 2) description 匹配
            desc = (p.get("description", "") or "").strip().lower()
            if desc:
                if desc[:80] in lower:
                    token_hits += 2
                else:
                    # 粗略关键词命中
                    desc_words = [w for w in desc.split() if len(w) > 4][:8]
                    token_hits += sum(1 for w in desc_words if w in lower)

            # 3) strategy name 匹配
            strategy_name = (p.get("strategy_name", "") or "").strip().lower()
            if strategy_name and strategy_name.replace("_", " ") in lower:
                token_hits += 2

            retrieval_score = self._safe_float(p.get("retrieval_score", 0.0))
            reliability_mean = self._safe_float(p.get("reliability_mean", 0.0))
            reliability = self._safe_float(p.get("reliability", 0.0))

            base = max(retrieval_score, 0.0) + max(reliability_mean, 0.0) + 0.5 * max(reliability, 0.0)

            # token_hits 越多，支持越强；但最终截断到 [0,1]
            score = 0.35 * base + 0.12 * token_hits
            best_score = max(best_score, score)

        return self._clamp01(best_score)

    def _copy_penalty(self, candidate_text: str, prototypes: List[Dict]) -> float:
        """
        防止直接把 prototype description 当答案抄进去。
        这是一个轻量启发式：
        - description 前缀整段出现在 candidate 里，惩罚较高
        - strategy_name / common_steps 原样大面积出现，也适度惩罚
        """
        if not self.enable_copy_penalty or not prototypes:
            return 0.0

        lower = (candidate_text or "").lower()
        penalties = []

        for p in prototypes:
            if not isinstance(p, dict):
                continue
            local_penalty = 0.0

            desc = (p.get("description", "") or "").strip().lower()
            if desc:
                if desc[:120] and desc[:120] in lower:
                    local_penalty = max(local_penalty, 1.0)
                elif desc[:60] and desc[:60] in lower:
                    local_penalty = max(local_penalty, 0.7)

            strategy_name = (p.get("strategy_name", "") or "").strip().lower()
            if strategy_name and strategy_name.replace("_", " ") in lower:
                local_penalty = max(local_penalty, 0.3)

            common_steps = p.get("common_steps", []) or []
            if not isinstance(common_steps, list):
                common_steps = [str(common_steps)]
            hit_steps = 0
            for step in common_steps:
                step = (step or "").strip().lower()
                if step and step[:50] in lower:
                    hit_steps += 1
            if common_steps:
                local_penalty = max(local_penalty, min(0.8, hit_steps / max(1, len(common_steps))))

            penalties.append(local_penalty)

        return self._clamp01(max(penalties) if penalties else 0.0)

    def _estimate_confidence(self, candidate_text: str, answer: Optional[str]) -> float:
        """
        这里不再写死 0.8 / 0.2，而是做一个很轻量的启发式 confidence：
        - 有解析出的 answer 基础分更高
        - 文本过短或过乱则降一点
        """
        text = (candidate_text or "").strip()
        if answer is None:
            return 0.2

        length = len(text)
        if length < 20:
            return 0.45
        if length < 80:
            return 0.65
        return 0.80

    def _answer_consistency(
        self,
        answer: Optional[str],
        auxiliary_samples: Optional[List[str]] = None,
    ) -> float:
        """
        如果有 auxiliary samples，就看多个 sample 的 answer 一致性；
        如果没有，就不要直接退化成 validity=1.0，而是给一个中性值。
        """
        if not auxiliary_samples:
            return 0.5 if answer is not None and str(answer).strip() else 0.0

        aux_answers = [safe_parse_answer(x) for x in auxiliary_samples]
        matches = sum(a == answer and a is not None for a in aux_answers)
        return self._clamp01(matches / max(1, len(auxiliary_samples)))

    def score_candidate(
        self,
        question: str,
        response_text: Optional[str] = None,
        candidate_text: Optional[str] = None,
        prototypes: Optional[List[Dict]] = None,
        ordered_prototypes: Optional[List[Dict]] = None,
        auxiliary_samples: Optional[List[str]] = None,
    ) -> Dict:
        """
        兼容两种调用方式：
        - score_candidate(question=..., candidate_text=..., prototypes=...)
        - score_candidate(question=..., response_text=..., ordered_prototypes=...)
        """
        text = response_text if response_text is not None else candidate_text
        if text is None:
            text = ""

        proto_list = ordered_prototypes if ordered_prototypes is not None else prototypes
        proto_list = proto_list or []

        answer = safe_parse_answer(text)
        validity = 1.0 if answer is not None else 0.0
        confidence = self._estimate_confidence(text, answer)
        answer_consistency = self._answer_consistency(answer, auxiliary_samples=auxiliary_samples)
        memory_support = self._memory_support(text, proto_list)

        if self.enable_ct:
            try:
                ct_raw = compute_ct_alignment(text, proto_list)
            except Exception:
                ct_raw = {}
            ct_info = {
                "coverage": self._clamp01(self._safe_float(ct_raw.get("coverage", 0.0))),
                "collapse_penalty": self._clamp01(self._safe_float(ct_raw.get("collapse_penalty", 0.0))),
                "ct_cost": self._clamp01(self._safe_float(ct_raw.get("ct_cost", 0.0))),
            }
        else:
            ct_info = {
                "coverage": 0.0,
                "collapse_penalty": 0.0,
                "ct_cost": 0.0,
            }

        copy_penalty = self._copy_penalty(text, proto_list)

        score = (
            self._weight("answer_consistency") * answer_consistency
            + self._weight("memory_support") * memory_support
            + self._weight("validity") * validity
            + self._weight("prototype_coverage") * ct_info["coverage"]
            + self._weight("confidence") * confidence
            - self._weight("copy_penalty") * copy_penalty
            - self._weight("collapse_penalty") * ct_info["collapse_penalty"]
            - self._weight("ct_cost") * ct_info["ct_cost"]
        )

        return {
            "score": float(score),
            "answer": answer,
            "breakdown": {
                "answer_consistency": float(answer_consistency),
                "memory_support": float(memory_support),
                "validity": float(validity),
                "prototype_coverage": float(ct_info["coverage"]),
                "confidence": float(confidence),
                "copy_penalty": float(copy_penalty),
                "collapse_penalty": float(ct_info["collapse_penalty"]),
                "ct_cost": float(ct_info["ct_cost"]),
            },
        }
