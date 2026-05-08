"""
Data API
"""
import json
import os

from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from prompts import (
    gsm8k_prompt,
    asdiv_aug_prompt,
    math_500_prompt,
    aime_prompt,
    strategy_qa_prompt,
    du_prompt,
)


def _safe_pick_split(ds, preferred_split: str, fallback_split: str):
    if hasattr(ds, "keys"):
        if preferred_split in ds:
            return ds[preferred_split]
        if fallback_split in ds:
            return ds[fallback_split]
        # 兜底：取第一个 split
        first_key = list(ds.keys())[0]
        return ds[first_key]
    return ds


def get_dataset(data_name_or_path, tokenizer, prompt_idx, split=None):
    """
    Args:
        data_name_or_path: dataset name or path
        tokenizer: tokenizer
        prompt_idx: which query prompt to use
        split: dataset split, e.g. train / test / validation
    Returns:
        dataset
    """
    data_name_lower = data_name_or_path.lower()
    split = split or "test"

    # ===== Load dataset =====
    if "gsm8k" in data_name_lower:
        try:
            dataset = _safe_pick_split(load_from_disk(data_name_or_path), split, "test")
        except Exception:
            dataset = load_dataset("openai/gsm8k", "socratic")[split if split in ["train", "test"] else "test"]
        question_col = "question"
        answer_col = "answer"

    elif "asdiv-aug" in data_name_lower:
        try:
            dataset = _safe_pick_split(load_from_disk(data_name_or_path), split, "test")
        except Exception:
            dataset = load_dataset("xuyige/ASDiv-Aug")[split if split in ["train", "test", "validation"] else "test"]
        question_col = "question"
        answer_col = "answer"

    elif "math-500" in data_name_lower:
        try:
            dataset = _safe_pick_split(load_from_disk(data_name_or_path), split, "test")
        except Exception:
            dataset = load_dataset("HuggingFaceH4/MATH-500")[split if split in ["test", "train"] else "test"]
        question_col = "problem"
        answer_col = "answer"

    elif "aime_2024" in data_name_lower:
        try:
            dataset = load_from_disk(data_name_or_path)
            if hasattr(dataset, "keys"):
                dataset = _safe_pick_split(dataset, split, "train")
        except Exception:
            dataset = load_dataset("Maxwell-Jia/AIME_2024")[split if split in ["train", "test"] else "train"]
        question_col = "Problem"
        answer_col = "Answer"

    elif "aime2025" in data_name_lower:
        try:
            dataset = load_from_disk(data_name_or_path)
            if hasattr(dataset, "keys"):
                dataset = _safe_pick_split(dataset, split, "test")
        except Exception:
            dataset = concatenate_datasets([
                load_dataset("opencompass/AIME2025", "AIME2025-I")["test"],
                load_dataset("opencompass/AIME2025", "AIME2025-II")["test"],
            ])
        question_col = "question"
        answer_col = "answer"

    elif "strategyqa" in data_name_lower:
        return get_strategyqa(tokenizer, prompt_idx)

    elif "date_understanding" in data_name_lower:
        ds = load_dataset("maveriq/bigbenchhard", "date_understanding")
        dataset = _safe_pick_split(ds, split, "train")
        question_col = "input"
        answer_col = "target"

    else:
        raise ValueError(f"Unsupported dataset: {data_name_or_path}")

    # ===== Preprocess dataset =====
    def preprocess_function(examples):
        prompt = []
        formatted = []
        answers = examples[answer_col]
        questions = examples[question_col]

        for q in questions:
            if "gsm8k" in data_name_lower:
                messages = gsm8k_prompt(q, prompt_idx)
            elif "asdiv-aug" in data_name_lower:
                messages = asdiv_aug_prompt(q, prompt_idx)
            elif "math-500" in data_name_lower:
                messages = math_500_prompt(q, prompt_idx)
            elif "aime" in data_name_lower:
                messages = aime_prompt(q, prompt_idx)
            elif "date_understanding" in data_name_lower:
                messages = du_prompt(q, prompt_idx)
            else:
                raise ValueError(f"Unsupported dataset: {data_name_or_path}")

            prompt.append(messages)
            formatted.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        if "aime" in data_name_lower and "2025" in data_name_lower:
            answers = [ans.replace("^\\circ", "").replace("^\circ", "") for ans in answers]

        if "date_understanding" in data_name_lower:
            # 原 target 类似 "(A)"，保留中间字母
            answers = [ans[1] if isinstance(ans, str) and len(ans) >= 2 else ans for ans in answers]

        return {
            "prompt": prompt,
            "formatted": formatted,
            "question": questions,
            "answer": answers,
        }

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=False,
        keep_in_memory=True,
    )
    return dataset


def get_strategyqa(tokenizer, prompt_idx):
    prompt = []
    formatted = []
    answers = []
    questions = []

    candidate_paths = [
        "strategyqa_train.json",
        os.path.join(os.path.dirname(__file__), "strategyqa_train.json"),
    ]

    file_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            file_path = p
            break

    if file_path is None:
        raise FileNotFoundError("strategyqa_train.json not found in current directory or project root.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for ins in data:
        q, a = ins["question"], ins["answer"]
        msg = strategy_qa_prompt(q, prompt_idx)
        questions.append(q)
        answers.append(a)
        prompt.append(msg)
        formatted.append(
            tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    return Dataset.from_dict({
        "prompt": prompt,
        "formatted": formatted,
        "question": questions,
        "answer": answers,
    })
