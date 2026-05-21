import inspect
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from memory_retriever import PrototypeRetriever
from memory_scorer import MemoryScorer
from prompts_memory import build_memory_guided_messages


DEFAULT_BREAKDOWN_KEYS = (
    "answer_consistency",
    "memory_support",
    "validity",
    "prototype_coverage",
    "confidence",
    "copy_penalty",
    "collapse_penalty",
    "ct_cost",
)

GROUP_SIZE_BONUS = 0.02


def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def _decode_new_tokens(tokenizer, outputs, prompt_len: int) -> str:
    seq = outputs[0][prompt_len:]
    return tokenizer.decode(seq, skip_special_tokens=True).strip()


def _build_retriever(prototype_path: str, vectorizer_path: str = "") -> PrototypeRetriever:
    """
    兼容不同 PrototypeRetriever 构造函数：
    - PrototypeRetriever(prototype_path)
    - PrototypeRetriever(prototype_path, vectorizer_path=...)
    """
    sig = inspect.signature(PrototypeRetriever.__init__)
    kwargs = {}
    if "vectorizer_path" in sig.parameters and vectorizer_path:
        kwargs["vectorizer_path"] = vectorizer_path
    return PrototypeRetriever(prototype_path, **kwargs)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _call_retrieve(retriever: PrototypeRetriever, question: str, top_k: int) -> List[Dict[str, Any]]:
    """
    兼容 retrieve(question, top_k=...)
    或 retrieve(question, k=...)
    """
    sig = inspect.signature(retriever.retrieve)
    kwargs = {}
    if "top_k" in sig.parameters:
        kwargs["top_k"] = top_k
    elif "k" in sig.parameters:
        kwargs["k"] = top_k
    return retriever.retrieve(question, **kwargs)


def _call_score_candidate(
    scorer: MemoryScorer,
    question: str,
    response_text: str,
    ordered_prototypes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    兼容不同 score_candidate 接口。
    """
    sig = inspect.signature(scorer.score_candidate)
    kwargs = {}

    if "question" in sig.parameters:
        kwargs["question"] = question
    if "response_text" in sig.parameters:
        kwargs["response_text"] = response_text
    elif "response" in sig.parameters:
        kwargs["response"] = response_text
    elif "text" in sig.parameters:
        kwargs["text"] = response_text

    if "ordered_prototypes" in sig.parameters:
        kwargs["ordered_prototypes"] = ordered_prototypes
    elif "prototypes" in sig.parameters:
        kwargs["prototypes"] = ordered_prototypes

    if "auxiliary_samples" in sig.parameters:
        kwargs["auxiliary_samples"] = None

    result = scorer.score_candidate(**kwargs)

    # 统一返回格式
    if isinstance(result, dict):
        if "score" not in result:
            raise ValueError("score_candidate() returned a dict without key 'score'.")
        result["score"] = _safe_float(result.get("score", 0.0))
        result.setdefault("answer", None)
        result["breakdown"] = _normalize_breakdown(result.get("breakdown", {}))
        return result

    raise ValueError("score_candidate() must return a dict containing at least {'score': ...}.")


def _rotate_prototypes(prototypes: List[Dict[str, Any]], offset: int) -> List[Dict[str, Any]]:
    if not prototypes:
        return []
    offset = offset % len(prototypes)
    return prototypes[offset:] + prototypes[:offset]


def _normalize_breakdown(breakdown: Optional[Dict[str, Any]]) -> Dict[str, float]:
    breakdown = breakdown if isinstance(breakdown, dict) else {}
    return {key: _safe_float(breakdown.get(key, 0.0)) for key in DEFAULT_BREAKDOWN_KEYS}


def _answer_group_key(candidate: Dict[str, Any]) -> str:
    answer = candidate.get("answer", None)
    if answer is not None and str(answer).strip():
        return str(answer).strip()
    return f"__no_answer_candidate_{candidate.get('candidate_idx', len(str(candidate)))}"


def _select_best_candidate_by_answer_group(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        groups.setdefault(_answer_group_key(candidate), []).append(candidate)

    group_records = []
    for group_key, members in groups.items():
        scores = [_safe_float(member.get("score", 0.0)) for member in members]
        best_member = max(
            members,
            key=lambda x: (
                _safe_float(x.get("score", 0.0)),
                -int(x.get("candidate_idx", 10**9)),
            ),
        )
        best_member_score = _safe_float(best_member.get("score", 0.0))

        # 改成“组内最高分 + 小的组大小奖励”
        group_score = best_member_score + GROUP_SIZE_BONUS * (len(members) - 1)

        group_records.append(
            {
                "group_key": group_key,
                "members": members,
                "score": float(group_score),
                "best_member_score": float(best_member_score),
                "best_candidate_idx": int(best_member.get("candidate_idx", 10**9)),
            }
        )

    best_group = max(
        group_records,
        key=lambda x: (
            x["score"],
            x["best_member_score"],
            -x["best_candidate_idx"],
        ),
    )
    return max(
        best_group["members"],
        key=lambda x: (
            _safe_float(x.get("score", 0.0)),
            -int(x.get("candidate_idx", 10**9)),
        ),
    )


def _build_generation_kwargs(
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    max_new_tokens = max(1, int(max_new_tokens))
    if temperature is None or temperature <= 0:
        return {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
        }
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "num_beams": 1,
    }


def generate_with_memory(
    tokenizer,
    model,
    question: str,
    data_name: str,
    prototype_path: str,
    top_k_prototypes: int,
    n_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    scorer: MemoryScorer,
    verbose: int = 0,
    vectorizer_path: str = "",
):
    """
    Returns:
        (
            best_response: str,
            best_score: float,
            best_candidate_idx: int,
            retrieved_prototypes: List[Dict],
            candidates: List[Dict],
            best_breakdown: Dict,
        )
    """
    if scorer is None:
        raise ValueError("scorer must not be None for memory_ltpo.")

    device = _get_model_device(model)

    try:
        retriever = _build_retriever(prototype_path=prototype_path, vectorizer_path=vectorizer_path)
        prototypes = _call_retrieve(retriever, question=question, top_k=top_k_prototypes)
    except Exception as exc:
        if verbose:
            print(f"[memory_ltpo] prototype retrieval disabled: {exc}")
        prototypes = []
    if prototypes is None:
        prototypes = []

    if verbose:
        print(f"[memory_ltpo] retrieved {len(prototypes)} prototypes")

    # 3) generate candidates
    candidates: List[Dict[str, Any]] = []
    gen_kwargs = _build_generation_kwargs(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    effective_candidates = max(1, int(n_candidates or 1))

    for cand_idx in range(effective_candidates):
        ordered = _rotate_prototypes(prototypes, cand_idx)

        messages = build_memory_guided_messages(
            question=question,
            retrieved_prototypes=ordered,
            data_name=data_name,
        )

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = _move_batch_to_device(inputs, device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                **gen_kwargs,
            )

        text = _decode_new_tokens(tokenizer, outputs, prompt_len)

        score_info = _call_score_candidate(
            scorer=scorer,
            question=question,
            response_text=text,
            ordered_prototypes=ordered,
        )

        candidate_record = {
            "candidate_idx": cand_idx,
            "response": text,
            "score": float(score_info["score"]),
            "answer": score_info.get("answer", None),
            "breakdown": _normalize_breakdown(score_info.get("breakdown", {})),
            "prototype_ids": [p.get("prototype_id", f"proto_{j}") for j, p in enumerate(ordered)],
        }
        candidates.append(candidate_record)

        if verbose > 1:
            print(
                f"[memory_ltpo] candidate={cand_idx} "
                f"score={candidate_record['score']:.4f} "
                f"answer={candidate_record['answer']}"
            )

    if not candidates:
        raise RuntimeError("No candidates generated in memory_ltpo.")

    # 4) select best answer group, then best candidate inside that group
    best = _select_best_candidate_by_answer_group(candidates)

    return (
        best["response"],
        best["score"],
        best["candidate_idx"],
        prototypes,
        candidates,
        best["breakdown"],
    )
