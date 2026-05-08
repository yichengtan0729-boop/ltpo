from typing import Dict, List


def _format_prototype_block(idx: int, proto: Dict) -> str:
    steps = proto.get("common_steps", []) or []
    if not isinstance(steps, list):
        steps = [str(steps)]
    steps = steps[:2]
    if steps:
        step_txt = "\n".join([f"  - {' '.join(str(s).split())[:140]}" for s in steps])
    else:
        step_txt = "  - adapt the reasoning pattern to the current problem"

    desc = (proto.get("description", "") or "").strip()
    desc = " ".join(desc.split())
    if "." in desc:
        desc = desc.split(".", 1)[0].strip() + "."
    if len(desc) > 160:
        desc = desc[:160].rsplit(" ", 1)[0].strip() + "..."
    if not desc:
        desc = "Use a short, adaptable reasoning pattern."

    strategy_name = proto.get("strategy_name", f"prototype_{idx}")

    block = (
        f"Prototype {idx}: {strategy_name}\n"
        f"Description: {desc}\n"
        f"Common steps:\n{step_txt}"
    )
    return block


def build_memory_guided_messages(question: str, retrieved_prototypes: List[Dict], data_name: str = "") -> List[Dict[str, str]]:
    bullet_blocks = []
    for idx, proto in enumerate(retrieved_prototypes, start=1):
        bullet_blocks.append(_format_prototype_block(idx, proto))

    proto_text = "\n\n".join(bullet_blocks) if bullet_blocks else "No prototype available."

    data_name_lower = (data_name or "").lower()
    if "strategyqa" in data_name_lower:
        answer_rule = "You must answer with Yes or No, and place the final answer in \\boxed{}."
    elif "date_understanding" in data_name_lower:
        answer_rule = "You must answer with one option among A, B, C, D, E, F, and place it in \\boxed{}."
    else:
        answer_rule = "You must place the final answer in \\boxed{}."

    system_content = (
        "You are solving a reasoning problem. "
        "You will be given several retrieved reasoning prototypes distilled from previous self-generated solutions. "
        "These are reasoning patterns, not answers to the current problem. "
        "Use them only as lightweight reasoning guidance."
    )

    user_content = (
        "Retrieved reasoning prototypes:\n\n"
        f"{proto_text}\n\n"
        "Instructions:\n"
        "- Do not copy historical answers.\n"
        "- These are reasoning patterns, not the current answer.\n"
        "- Adapt the reasoning to the current problem.\n"
        "- Prefer the most stable and well-supported reasoning path.\n"
        f"- {answer_rule}\n\n"
        f"Problem: {question}"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
