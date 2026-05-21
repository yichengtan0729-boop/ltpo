DEFAULT_ROLES = [
    "understand_extract",
    "plan_retrieve",
    "infer_compute",
    "verify_correct",
]


def infer_scenario(question: str, dataset_name: str = "") -> str:
    name = (dataset_name or "").lower()
    if "gsm8k" in name:
        return "math_arithmetic"
    if "aime" in name:
        return "math_competition"
    if "math" in name:
        return "math_general"
    if "strategyqa" in name:
        return "commonsense_binary"
    if "date_understanding" in name:
        return "date_reasoning"
    if "cruxeval" in name or "code" in name:
        return "code_reasoning"
    return "general_reasoning"
