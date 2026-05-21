import argparse
import inspect
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import get_dataset
from extract_judge_answer import extract_answer, extract_true_answer, judge_answer
from ltpo import generate as ltpo_generate
from memory_builder import build_memory_bank
from memory_ltpo import generate_with_memory
from memory_scorer import MemoryScorer
from prototype_builder import build_prototypes
from reward import RewardModel
from utils_io import ensure_dir, safe_name_from_path


huggingface_token = os.environ.get("HUGGING_FACE_TOKEN", None)


def parse_args():
    parser = argparse.ArgumentParser(description="LTPO + Memory pipeline")

    # common
    parser.add_argument("--method", type=str, default="ltpo",
                        choices=["baseline", "ltpo", "memory_ltpo", "build_memory", "build_prototypes"])
    parser.add_argument("--dataset", type=str, default="openai/gsm8k", help="Dataset to evaluate")
    parser.add_argument("--dataset_split", type=str, default="test", help="Split for evaluation")
    parser.add_argument("--memory_dataset", type=str, default="", help="Dataset to build memory from")
    parser.add_argument("--memory_split", type=str, default="train", help="Split to build memory from")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or HF name of the model")
    parser.add_argument("--output_dir", type=str, default="./output", help="Path to the output directory")
    parser.add_argument("--start_data_idx", type=int, default=0)
    parser.add_argument("--end_data_idx", type=int, default=-1, help="-1 means run to the end")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--solver_prompt_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # original LTPO args
    parser.add_argument("--num_thought_tokens", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--sigma_decay", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--max_num_steps", type=int, default=10)
    parser.add_argument("--reward_threshold", type=float, default=-1)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--disable_conf_reward", action="store_true")
    parser.add_argument("--disable_best_reward", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt_suffix", type=str, default="")
    parser.add_argument("--use_auto_grad", action="store_true")

    # memory pipeline args
    parser.add_argument("--memory_dir", type=str, default="./output/memories")
    parser.add_argument("--prototype_dir", type=str, default="./output/prototypes")
    parser.add_argument("--memory_output_path", type=str, default="")
    parser.add_argument("--prototype_path", type=str, default="")
    parser.add_argument("--vectorizer_path", type=str, default="")
    parser.add_argument("--n_memory_samples", type=int, default=5)
    parser.add_argument("--memory_temperature", type=float, default=0.3)
    parser.add_argument("--memory_top_p", type=float, default=0.9)
    parser.add_argument("--n_prototypes", type=int, default=16)
    parser.add_argument("--top_examples_per_prototype", type=int, default=3)
    parser.add_argument("--prototype_max_features", type=int, default=4096)
    parser.add_argument("--top_k_prototypes", type=int, default=2)
    parser.add_argument("--n_candidates", type=int, default=2)
    parser.add_argument("--min_memory_reliability", type=float, default=0.25)
    parser.add_argument("--disable_ct", action="store_true")
    parser.add_argument("--disable_copy_penalty", action="store_true")

    # misc
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--disable_save_logistics", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
    "answer_consistency": 0.55,
    "memory_support": 0.20,
    "validity": 0.25,
    "prototype_coverage": 0.00,
    "confidence": 0.00,
    "copy_penalty": 0.02,
    "collapse_penalty": 0.00,
    "ct_cost": 0.00,
}


def _resolve_dtype(dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    # auto
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _resolve_device(device_name: str) -> str:
    requested = (device_name or "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[main] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return device_name


def _move_batch_to_device(batch, device):
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    return batch


def _load_model_and_tokenizer(args):
    device = _resolve_device(args.device)
    args.device = device
    dtype = _resolve_dtype(args.dtype)
    if device == "cpu" and dtype != torch.float32:
        dtype = torch.float32

    # 单卡更稳：直接加载后 model.to(device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        token=huggingface_token,
    )
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=huggingface_token)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _build_output_dir(args) -> str:
    model_name = safe_name_from_path(args.model_name_or_path)
    data_name = safe_name_from_path(args.dataset)
    ensure_dir(args.output_dir)

    if args.method == "baseline":
        return os.path.join(args.output_dir, f"{model_name}-{data_name}-baseline-prompt{args.solver_prompt_idx}")

    if args.method == "ltpo":
        conf_suffix = "" if args.disable_conf_reward else "-conf"
        return os.path.join(
            args.output_dir,
            f"{model_name}-{data_name}-tokens{args.num_thought_tokens}-lr{args.lr}-sigma{args.sigma}-sigdecay{args.sigma_decay}{conf_suffix}",
        )

    if args.method == "memory_ltpo":
        return os.path.join(
            args.output_dir,
            f"{model_name}-{data_name}-memoryltpo-topk{args.top_k_prototypes}-cand{args.n_candidates}",
        )

    return args.output_dir


def _call_generate_with_memory(args, tokenizer, model, question, scorer):
    kwargs = {
        "tokenizer": tokenizer,
        "model": model,
        "question": question,
        "data_name": args.dataset,
        "prototype_path": args.prototype_path,
        "top_k_prototypes": args.top_k_prototypes,
        "n_candidates": args.n_candidates,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.memory_temperature,
        "top_p": args.memory_top_p,
        "scorer": scorer,
        "verbose": args.verbose,
    }
    if args.vectorizer_path:
        kwargs["vectorizer_path"] = args.vectorizer_path

    sig = inspect.signature(generate_with_memory)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return generate_with_memory(**filtered)


def run_build_memory(args):
    model, tokenizer = _load_model_and_tokenizer(args)

    dataset_name = args.memory_dataset if args.memory_dataset else args.dataset
    split_name = args.memory_split if args.memory_split else "train"

    dataset = get_dataset(
        dataset_name,
        tokenizer=tokenizer,
        prompt_idx=args.solver_prompt_idx,
        split=split_name,
    )
    path = build_memory_bank(args, model, tokenizer, dataset)
    print(f"[build_memory] saved to: {path}")


def run_build_prototypes(args):
    result = build_prototypes(args)
    print(f'[build_prototypes] prototype_path={result.get("prototype_path", "")}')
    print(f'[build_prototypes] vectorizer_path={result.get("vectorizer_path", "")}')


def main(args):
    if args.seed is not None:
        set_seed(args.seed)

    if args.method == "build_memory":
        run_build_memory(args)
        return

    if args.method == "build_prototypes":
        run_build_prototypes(args)
        return

    model, tokenizer = _load_model_and_tokenizer(args)
    dataset = get_dataset(
        args.dataset,
        tokenizer=tokenizer,
        prompt_idx=args.solver_prompt_idx,
        split=args.dataset_split,
    )

    reward_model: Optional[RewardModel] = None
    scorer: Optional[MemoryScorer] = None

    if args.method == "ltpo":
        reward_model = RewardModel(
            model=model,
            tokenizer=tokenizer,
            num_thought_tokens=args.num_thought_tokens,
            device=args.device,
            model_name=args.model_name_or_path,
        )
    elif args.method == "memory_ltpo":
        scorer = MemoryScorer(
            weights=DEFAULT_SCORE_WEIGHTS,
            enable_ct=not args.disable_ct,
            enable_copy_penalty=not args.disable_copy_penalty,
        )

    total = 0
    correct = 0
    entries = []
    output_dir = _build_output_dir(args)
    ensure_dir(output_dir)

    start_data_idx = max(0, args.start_data_idx)
    end_data_idx = len(dataset) if args.end_data_idx is None or args.end_data_idx < 0 else min(args.end_data_idx, len(dataset))

    if args.verbose:
        print(f"Running method={args.method} on [{start_data_idx}, {end_data_idx})")
        print(f"Output dir: {output_dir}")

    for i in tqdm(range(start_data_idx, end_data_idx)):
        example = dataset[i]
        question = example["question"]
        true_answer = extract_true_answer(example["answer"], name=args.dataset)
        if true_answer is None:
            continue

        if args.method == "baseline":
            inputs = tokenizer.apply_chat_template(
                example["prompt"],
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = _move_batch_to_device(inputs, model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            best_reward = None
            best_reward_step = None
            extra = {}

        elif args.method == "ltpo":
            output, best_reward, best_reward_step = ltpo_generate(
                tokenizer=tokenizer,
                model=model,
                reward_model=reward_model,
                question=question,
                num_thought_tokens=args.num_thought_tokens,
                max_rl_steps=args.max_num_steps,
                max_new_tokens=args.max_new_tokens,
                reward_threshold=args.reward_threshold,
                lr=args.lr,
                sigma=args.sigma,
                sigma_decay=args.sigma_decay,
                use_auto_grad=args.use_auto_grad,
                disable_conf_reward=args.disable_conf_reward,
                disable_best_reward=args.disable_best_reward,
                data_name=args.dataset,
                model_name=args.model_name_or_path,
                verbose=args.verbose,
                top_k=args.top_k,
            )
            extra = {}

        elif args.method == "memory_ltpo":
            if not args.prototype_path:
                model_name = safe_name_from_path(args.model_name_or_path)
                data_name = safe_name_from_path(args.dataset)
                args.prototype_path = os.path.join(args.prototype_dir, f"{model_name}-{data_name}-prototypes.json")

            result = _call_generate_with_memory(
                args=args,
                tokenizer=tokenizer,
                model=model,
                question=question,
                scorer=scorer,
            )

            # 兼容不同返回格式
            if isinstance(result, tuple) and len(result) == 6:
                output, best_reward, best_reward_step, retrieved_prototypes, candidates, score_breakdown = result
            elif isinstance(result, tuple) and len(result) == 4:
                output, retrieved_prototypes, candidates, score_breakdown = result
                best_reward = None
                best_reward_step = None
            else:
                raise ValueError("Unsupported return format from generate_with_memory().")

            extra = {
                "retrieved_prototypes": retrieved_prototypes,
                "candidate_responses": candidates,
                "score_breakdown": score_breakdown,
            }

        else:
            raise ValueError(f"Unsupported method: {args.method}")

        answer = extract_answer(
            output,
            data_name=args.dataset,
            prompt_idx=args.solver_prompt_idx,
            model_name=args.model_name_or_path,
        )

        is_correct = False
        if answer is not None:
            is_correct = judge_answer(
                output,
                true_answer,
                data_name=args.dataset,
                prompt_idx=args.solver_prompt_idx,
            )

        correct += int(is_correct)
        total += 1

        entry = {
            "data_idx": i,
            "question": question,
            "response": output,
            "answer": answer,
            "true_answer": true_answer,
            "is_correct": is_correct,
            "best_reward": best_reward,
            "best_reward_step": best_reward_step,
            "method": args.method,
            **extra,
        }
        entries.append(entry)

        if not args.disable_save_logistics:
            torch.save(
                {
                    "start_idx": i + 1,
                    "total": total,
                    "correct": correct,
                    "entries": entries,
                    "method": args.method,
                },
                os.path.join(output_dir, "logistics.pt"),
            )

        if args.verbose:
            print(f"[idx={i}] answer={answer} true={true_answer} correct={is_correct} acc={correct / max(1, total):.4f}")
            if args.method == "memory_ltpo" and args.verbose > 1:
                print(f"[memory] score breakdown: {entry.get('score_breakdown', {})}")

    print(f"Final accuracy: {correct}/{total} = {correct / max(1, total):.4f}")


if __name__ == "__main__":
    args = parse_args()
    for arg in vars(args):
        print(f"-- {arg}: {getattr(args, arg)}")
    main(args)
