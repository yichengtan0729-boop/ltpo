import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from extract_judge_answer import extract_answer, extract_true_answer, judge_answer
from memory_builder import (
    _build_generation_kwargs,
    _compute_reliability,
    _decode_new_tokens,
    _majority_answer,
    _move_batch_to_device,
)
from scenario_router import infer_scenario
from step_parser import parse_steps
from utils_io import append_jsonl, ensure_dir, safe_name_from_path
from utils_parse import safe_parse_answer


def _build_step_memory_path(args) -> str:
    if getattr(args, "step_memory_output_path", None):
        return args.step_memory_output_path

    data_name = safe_name_from_path(getattr(args, "memory_dataset", None) or args.dataset)
    model_name = safe_name_from_path(args.model_name_or_path)
    memory_dir = args.step_memory_dir or os.path.join(args.output_dir, "step_memories")
    return os.path.join(memory_dir, f"{model_name}-{data_name}-step-memory.jsonl")


def _safe_true_answer(example: Dict[str, Any], dataset_name: str) -> Optional[str]:
    try:
        answer = example.get("answer", None)
        if answer is None:
            return None
        return extract_true_answer(answer, name=dataset_name)
    except Exception:
        raw = example.get("answer", None)
        return str(raw).strip() if raw is not None and str(raw).strip() else None


def _safe_parse_response(text: str, args) -> Optional[str]:
    try:
        ans = extract_answer(
            text,
            data_name=getattr(args, "memory_dataset", None) or args.dataset,
            prompt_idx=getattr(args, "solver_prompt_idx", 0),
            model_name=getattr(args, "model_name_or_path", ""),
        )
        if ans is not None and str(ans).strip():
            return str(ans).strip()
    except Exception:
        pass
    return safe_parse_answer(text)


def _safe_answer_match(
    response: str,
    parsed_answer: Optional[str],
    true_answer: Optional[str],
    args,
) -> bool:
    if parsed_answer is None or true_answer is None:
        return False
    data_name = getattr(args, "memory_dataset", None) or args.dataset
    try:
        return bool(
            judge_answer(
                response,
                true_answer,
                data_name=data_name,
                extract=True,
                prompt_idx=getattr(args, "solver_prompt_idx", 0),
            )
        )
    except Exception:
        try:
            return bool(
                judge_answer(
                    parsed_answer,
                    true_answer,
                    data_name=data_name,
                    extract=False,
                    prompt_idx=getattr(args, "solver_prompt_idx", 0),
                )
            )
        except Exception:
            return str(parsed_answer).strip().lower() == str(true_answer).strip().lower()


def _trace_failure_reliability(response: str) -> float:
    length_sanity = 1.0 if 32 <= len(response or "") <= 6000 else 0.5
    return float(1.0 * length_sanity)


def _step_validity(step_text: str) -> float:
    return 1.0 if len((step_text or "").strip()) >= 8 else 0.5


def _shorten(text: str, max_chars: int = 180) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def _classify_traces(
    responses: List[str],
    parsed_answers: List[Optional[str]],
    true_answer: Optional[str],
    args,
) -> Tuple[Optional[str], List[int], List[int]]:
    if true_answer is not None and str(true_answer).strip():
        verified = [
            idx for idx, ans in enumerate(parsed_answers)
            if ans is not None and _safe_answer_match(responses[idx], ans, true_answer, args)
        ]
        failures = [
            idx for idx, ans in enumerate(parsed_answers)
            if ans is not None and idx not in set(verified)
        ]
        return str(true_answer), verified, failures

    chosen_answer = _majority_answer(parsed_answers)
    if chosen_answer is None:
        return None, [], []
    verified = [idx for idx, ans in enumerate(parsed_answers) if ans == chosen_answer]
    failures = [idx for idx, ans in enumerate(parsed_answers) if ans is not None and ans != chosen_answer]
    return chosen_answer, verified, failures


def _append_step_entries(
    path: str,
    data_name: str,
    data_idx: int,
    question: str,
    scenario: str,
    response: str,
    parsed_answer: Optional[str],
    chosen_answer: Optional[str],
    true_answer: Optional[str],
    trace_reliability: float,
    trace_id: str,
    memory_type: str,
    failure_reason: str,
    max_steps: int,
) -> int:
    steps = parse_steps(question, response, dataset_name=data_name, max_steps=max_steps)
    if not steps:
        return 0

    is_failure = memory_type == "failure"
    is_correction = memory_type == "correction"
    written = 0
    for step in steps:
        step_text = step.get("step_text", "")
        step_validity = _step_validity(step_text)
        step_reliability = float(trace_reliability * step_validity)
        item = {
            "id": f"{data_name}_{data_idx}_trace_{trace_id}_step_{step.get('step_id', written)}",
            "data_idx": data_idx,
            "question": question,
            "scenario": scenario,
            "trace_id": f"{data_name}_{data_idx}_trace_{trace_id}",
            "step_id": int(step.get("step_id", written)),
            "role": step.get("role", ""),
            "step_type": step.get("step_type", "general_reasoning"),
            "step_text": step_text,
            "parsed_answer": parsed_answer,
            "chosen_answer": chosen_answer,
            "true_answer": true_answer,
            "trace_reliability": float(trace_reliability),
            "step_reliability": float(step_reliability),
            "embedding_text": f"{scenario} {step.get('role', '')} {step.get('step_type', '')} {question} {step_text}",
            "memory_type": memory_type,
            "is_failure": bool(is_failure),
            "is_correction": bool(is_correction),
            "failure_reason": failure_reason if is_failure else "",
        }
        append_jsonl(path, item)
        written += 1
    return written


def _append_correction_entry(
    path: str,
    data_name: str,
    data_idx: int,
    question: str,
    scenario: str,
    failure_response: str,
    correct_response: str,
    chosen_answer: Optional[str],
    true_answer: Optional[str],
    reliability: float,
    max_steps: int,
) -> None:
    try:
        fail_steps = parse_steps(question, failure_response, dataset_name=data_name, max_steps=max_steps)
        ok_steps = parse_steps(question, correct_response, dataset_name=data_name, max_steps=max_steps)
        if not fail_steps or not ok_steps:
            return
        wrong_summary = _shorten(" ".join(s.get("step_text", "") for s in fail_steps[:2]))
        correct_summary = _shorten(" ".join(s.get("step_text", "") for s in ok_steps[:2]))
        step_text = f"Failure pattern: {wrong_summary}. Correct pattern: {correct_summary}."
        item = {
            "id": f"{data_name}_{data_idx}_correction_step_0",
            "data_idx": data_idx,
            "question": question,
            "scenario": scenario,
            "trace_id": f"{data_name}_{data_idx}_correction_0",
            "step_id": 0,
            "role": "verify_correct",
            "step_type": "correction",
            "step_text": step_text,
            "parsed_answer": None,
            "chosen_answer": chosen_answer,
            "true_answer": true_answer,
            "trace_reliability": float(reliability),
            "step_reliability": float(reliability * _step_validity(step_text)),
            "embedding_text": f"{scenario} verify_correct correction {question} {step_text}",
            "memory_type": "correction",
            "is_failure": False,
            "is_correction": True,
            "failure_reason": "",
        }
        append_jsonl(path, item)
    except Exception:
        return


def build_step_memory_bank(args, model, tokenizer, dataset) -> str:
    data_name_raw = getattr(args, "memory_dataset", None) or args.dataset
    data_name = safe_name_from_path(data_name_raw)
    step_memory_path = _build_step_memory_path(args)
    ensure_dir(os.path.dirname(step_memory_path))

    start_data_idx = max(0, int(getattr(args, "start_data_idx", 0)))
    if getattr(args, "end_data_idx", -1) is None or args.end_data_idx < 0:
        end_data_idx = len(dataset)
    else:
        end_data_idx = min(int(args.end_data_idx), len(dataset))

    if start_data_idx == 0 and not getattr(args, "resume", False):
        open(step_memory_path, "w", encoding="utf-8").close()

    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    device = next(model.parameters()).device
    gen_kwargs = _build_generation_kwargs(
        max_new_tokens=getattr(args, "max_new_tokens", 1024),
        temperature=getattr(args, "memory_temperature", 0.3),
        top_p=getattr(args, "memory_top_p", 0.9),
    )
    n_memory_samples = max(1, int(getattr(args, "n_memory_samples", 5)))
    min_verified_reliability = float(getattr(args, "min_memory_reliability", 0.25))
    min_failure_reliability = float(getattr(args, "min_failure_reliability", 0.25))
    max_steps = max(1, int(getattr(args, "num_step_roles", 4)))

    for i in tqdm(range(start_data_idx, end_data_idx), desc="Building step memory"):
        try:
            example = dataset[i]
            question = example["question"]
            prompt = example.get("prompt", [{"role": "user", "content": question}])
            scenario = infer_scenario(question, data_name_raw)
            true_answer = _safe_true_answer(example, data_name_raw)

            inputs = tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = _move_batch_to_device(inputs, device)
            prompt_len = inputs["input_ids"].shape[1]

            responses: List[str] = []
            parsed_answers: List[Optional[str]] = []
            with torch.inference_mode():
                for _ in range(n_memory_samples):
                    outputs = model.generate(**inputs, **gen_kwargs)
                    text = _decode_new_tokens(tokenizer, outputs, prompt_len)
                    responses.append(text)
                    parsed_answers.append(_safe_parse_response(text, args))

            chosen_answer, verified_idxs, failure_idxs = _classify_traces(
                responses=responses,
                parsed_answers=parsed_answers,
                true_answer=true_answer,
                args=args,
            )
            stats = _compute_reliability(parsed_answers, chosen_answer, responses)

            wrote_verified = False
            for trace_idx in verified_idxs:
                trace_reliability = float(stats.get("reliability", 0.0))
                if trace_reliability < min_verified_reliability:
                    continue
                written = _append_step_entries(
                    path=step_memory_path,
                    data_name=data_name,
                    data_idx=i,
                    question=question,
                    scenario=scenario,
                    response=responses[trace_idx],
                    parsed_answer=parsed_answers[trace_idx],
                    chosen_answer=chosen_answer,
                    true_answer=true_answer,
                    trace_reliability=trace_reliability,
                    trace_id=str(trace_idx),
                    memory_type="verified_solution",
                    failure_reason="",
                    max_steps=max_steps,
                )
                wrote_verified = wrote_verified or written > 0

            wrote_failure = False
            for trace_idx in failure_idxs:
                trace_reliability = _trace_failure_reliability(responses[trace_idx])
                if trace_reliability < min_failure_reliability:
                    continue
                written = _append_step_entries(
                    path=step_memory_path,
                    data_name=data_name,
                    data_idx=i,
                    question=question,
                    scenario=scenario,
                    response=responses[trace_idx],
                    parsed_answer=parsed_answers[trace_idx],
                    chosen_answer=chosen_answer,
                    true_answer=true_answer,
                    trace_reliability=trace_reliability,
                    trace_id=str(trace_idx),
                    memory_type="failure",
                    failure_reason="answer_mismatch",
                    max_steps=max_steps,
                )
                wrote_failure = wrote_failure or written > 0

            if wrote_verified and wrote_failure and verified_idxs and failure_idxs:
                correction_reliability = max(float(stats.get("reliability", 0.0)), min_failure_reliability)
                _append_correction_entry(
                    path=step_memory_path,
                    data_name=data_name,
                    data_idx=i,
                    question=question,
                    scenario=scenario,
                    failure_response=responses[failure_idxs[0]],
                    correct_response=responses[verified_idxs[0]],
                    chosen_answer=chosen_answer,
                    true_answer=true_answer,
                    reliability=correction_reliability,
                    max_steps=max_steps,
                )
        except Exception as exc:
            if getattr(args, "verbose", 0):
                print(f"[build_step_memory] skipped idx={i}: {exc}")
            continue

    return step_memory_path
