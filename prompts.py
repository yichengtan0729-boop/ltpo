"""
Solver prompts for different math / reasoning datasets.
"""


def _boxed_prompt(q):
    return [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": q},
    ]


def _json_math_prompt(q, extra_note=""):
    return [
        {"role": "system", "content": "You are a precise math question solver. Solve this math problem."},
        {"role": "user", "content":
            f"QUESTION: {q}\n"
            "Let's think step by step. "
            "Please provide your thought process and your final answer separately and respond in json format "
            "containing the keys \"thought process\" and \"final answer\". "
            "For example, your response should be "
            "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
            f"{extra_note}"
        },
    ]


def _no_cot_boxed_prompt(q):
    return [
        {"role": "system", "content": "Please put your final answer within \\boxed{}."},
        {"role": "user", "content": q},
    ]


def gsm8k_prompt(q, prompt_idx=0):
    prompt = [
        _boxed_prompt(q),
        _json_math_prompt(
            q,
            extra_note="Note that the final answer should be pure numbers, not calculation formulas, and without any units or explanation."
        ),
        _no_cot_boxed_prompt(q),
    ]
    return prompt[prompt_idx]


def asdiv_aug_prompt(q, prompt_idx=0):
    prompt = [
        _boxed_prompt(q),
        _json_math_prompt(
            q,
            extra_note="Note that the final answer should be pure numbers, not calculation formulas, and without any units or explanation."
        ),
        _no_cot_boxed_prompt(q),
    ]
    return prompt[prompt_idx]


def math_500_prompt(q, prompt_idx=0):
    prompt = [
        _boxed_prompt(q),
        _json_math_prompt(q),
        _no_cot_boxed_prompt(q),
    ]
    return prompt[prompt_idx]


def aime_prompt(q, prompt_idx=0):
    prompt = [
        _boxed_prompt(q),
        _json_math_prompt(
            q,
            extra_note="Note that the final answer should be pure numbers, not calculation formulas, and without any units or explanation."
        ),
        _no_cot_boxed_prompt(q),
    ]
    return prompt[prompt_idx]


def strategy_qa_prompt(q, prompt_idx=0):
    prompt = [
        [
            {"role": "system", "content": "Please reason step by step, and answer the following question with `Yes` or `No`."},
            {"role": "user", "content": q},
        ],
        [
            {"role": "system", "content": "You are required to answer the following question with `Yes` or `No`."},
            {"role": "user", "content":
                f"QUESTION: {q}\n"
                "Let's think step by step. "
                "Please provide your thought process and your final answer separately and respond in json format "
                "containing the keys \"thought process\" and \"final answer\". "
                "For example, your response should be "
                "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                "Note that the final answer should be pure `Yes` or `No`, without any details or explanation."
            },
        ],
        [
            {"role": "system", "content": "You are required to answer the following question with `Yes` or `No`."},
            {"role": "user", "content": q},
        ],
    ]
    return prompt[prompt_idx]


def du_prompt(q, prompt_idx=0):
    boxed_sys = (
        "Please reason step by step, and put your final answer within \\boxed{}. "
        "In this multiple-choice problem, the available options are A, B, C, D, E, and F. "
        "Only one letter from A to F is accepted in the answer span."
    )
    json_sys = (
        "You are a precise question solver. Solve this multiple-choice problem. "
        "The available options are A, B, C, D, E, and F. "
        "Only one letter from A to F is accepted in the final answer."
    )

    prompt = [
        [
            {"role": "system", "content": boxed_sys},
            {"role": "user", "content": q},
        ],
        [
            {"role": "system", "content": json_sys},
            {"role": "user", "content":
                f"QUESTION: {q}\n"
                "Let's think step by step. "
                "Please provide your thought process and your final answer separately and respond in json format "
                "containing the keys \"thought process\" and \"final answer\". "
                "For example, your response should be "
                "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                "Note that the final answer should be a pure option letter, without formulas, details, or explanation."
            },
        ],
        [
            {"role": "system", "content": boxed_sys},
            {"role": "user", "content": q},
        ],
    ]
    return prompt[prompt_idx]