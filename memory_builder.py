import os
from collections import Counter
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from utils_io import append_jsonl, ensure_dir, safe_name_from_path
from utils_parse import safe_parse_answer


def _decode_new_tokens(tokenizer, outputs, prompt_len: int) -> str:
    seq = outputs[0][prompt_len:]
    return tokenizer.decode(seq, skip_special_tokens=True).strip()


def _majority_answer(answers: List[Optional[str]]) -> Optional[str]:
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def _compute_reliability(
    parsed_answers: List[Optional[str]],
    chosen_answer: Optional[str],
    responses: List[str],
) -> Dict[str, float]:
    n = max(1, len(responses))
    valid_answers = [a for a in parsed_answers if a is not None]

    answer_consistency = 0.0
    if chosen_answer is not None:
        answer_consistency = sum(a == chosen_answer for a in parsed_answers) / n

    validity = len(valid_answers) / n
    avg_len = sum(len(r) for r in responses) / n
    length_sanity = 1.0 if 32 <= avg_len <= 6000 else 0.5

    reliability = 0.55 * answer_consistency + 0.30 * validity + 0.15 * length_sanity

    return {
        "answer_consistency": float(answer_consistency),
        "validity": float(validity),
        "length_sanity": float(length_sanity),
        "reliability": float(reliability),
    }


def _move_batch_to_device(batch, device):
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    return batch


def _build_generation_kwargs(max_new_tokens: int, temperature: float, top_p: float) -> Dict[str, Any]:
    kwargs = {
        "max_new_tokens": max(1, int(max_new_tokens)),
        "num_beams": 1,
    }
    if temperature is None or temperature <= 0:
        kwargs["do_sample"] = False
        return kwargs

    kwargs.update({
        "do_sample": True,
        "temperature": float(temperature),
        "top_p": float(top_p),
    })
    return kwargs


def _build_memory_path(args) -> str:
    if getattr(args, "memory_output_path", None):
        return args.memory_output_path

    data_name = safe_name_from_path(getattr(args, "memory_dataset", None) or args.dataset)
    model_name = safe_name_from_path(args.model_name_or_path)
    memory_dir = args.memory_dir or os.path.join(args.output_dir, "memories")
    return os.path.join(memory_dir, f"{model_name}-{data_name}-memory.jsonl")


def build_memory_bank(args, model, tokenizer, dataset: List[Dict[str, Any]]) -> str:
    data_name = safe_name_from_path(getattr(args, "memory_dataset", None) or args.dataset)
    memory_path = _build_memory_path(args)
    ensure_dir(os.path.dirname(memory_path))

    start_data_idx = max(0, args.start_data_idx)
    if getattr(args, "end_data_idx", -1) is None or args.end_data_idx < 0:
        end_data_idx = len(dataset)
    else:
        end_data_idx = min(args.end_data_idx, len(dataset))

    device = next(model.parameters()).device
    if start_data_idx == 0 and not getattr(args, "resume", False):
        open(memory_path, "w", encoding="utf-8").close()

    gen_kwargs = _build_generation_kwargs(
        max_new_tokens=args.max_new_tokens,
        temperature=args.memory_temperature,
        top_p=args.memory_top_p,
    )
    n_memory_samples = max(1, int(getattr(args, "n_memory_samples", 5)))
    min_reliability = float(getattr(args, "min_memory_reliability", 0.25))

    for i in tqdm(range(start_data_idx, end_data_idx), desc="Building memory"):
        example = dataset[i]
        question = example["question"]
        prompt = example["prompt"]

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
                outputs = model.generate(
                    **inputs,
                    **gen_kwargs,
                )
                text = _decode_new_tokens(tokenizer, outputs, prompt_len)
                responses.append(text)
                parsed_answers.append(safe_parse_answer(text))

        chosen_answer = _majority_answer(parsed_answers)

        chosen_idx = 0
        if chosen_answer is not None:
            for j, ans in enumerate(parsed_answers):
                if ans == chosen_answer:
                    chosen_idx = j
                    break

        stats = _compute_reliability(parsed_answers, chosen_answer, responses)
        if stats["reliability"] < min_reliability:
            continue

        item = {
            "id": f"{data_name}_{i}",
            "data_idx": i,
            "question": question,
            "prompt": prompt,
            "responses": responses,
            "parsed_answers": parsed_answers,
            "chosen_rationale": responses[chosen_idx],
            "chosen_answer": chosen_answer,
            "embedding_text": f"{question}\n\n{responses[chosen_idx]}",
            **stats,
        }
        append_jsonl(memory_path, item)

    return memory_path
