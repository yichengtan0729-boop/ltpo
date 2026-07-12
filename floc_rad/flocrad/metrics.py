from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .schema import Edit, FindingState, apply_edits, canonical_edits, state_from_json


def _f1(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    beta2 = beta * beta
    denom = (1 + beta2) * tp + beta2 * fn + fp
    return 0.0 if denom == 0 else (1 + beta2) * tp / denom


def evaluate_predictions(rows: Sequence[Mapping[str, object]], beta: float = 0.5) -> Dict[str, object]:
    tp = fp = fn = 0
    exact = 0
    clean = 0
    clean_false_edits = 0
    preserved = 0
    preserve_total = 0
    by_k = defaultdict(lambda: {"n": 0, "exact": 0})
    by_operator = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for row in rows:
        gold = {Edit.from_dict(item).key: Edit.from_dict(item) for item in row.get("gold_edits", [])}
        pred = {Edit.from_dict(item).key: Edit.from_dict(item) for item in row.get("pred_edits", [])}
        gold_keys, pred_keys = set(gold), set(pred)
        current_tp = len(gold_keys & pred_keys)
        current_fp = len(pred_keys - gold_keys)
        current_fn = len(gold_keys - pred_keys)
        tp += current_tp
        fp += current_fp
        fn += current_fn
        is_exact = gold_keys == pred_keys
        exact += int(is_exact)
        k = int(row.get("k", len(gold_keys)))
        by_k[k]["n"] += 1
        by_k[k]["exact"] += int(is_exact)

        if k == 0:
            clean += 1
            clean_false_edits += int(bool(pred_keys))

        for key in gold_keys | pred_keys:
            op = (gold.get(key) or pred.get(key)).op
            by_operator[op]["tp"] += int(key in gold_keys and key in pred_keys)
            by_operator[op]["fp"] += int(key not in gold_keys and key in pred_keys)
            by_operator[op]["fn"] += int(key in gold_keys and key not in pred_keys)

        corrupted = state_from_json(str(row["corrupted_state"]))
        target = state_from_json(str(row["gold_state"]))
        corrected = apply_edits(corrupted, list(pred.values()))
        gold_targets = {(edit.finding, edit.op) for edit in gold.values()}
        for finding, target_slot in target.items():
            corr_slot = corrupted[finding]
            pred_slot = corrected[finding]
            for attribute in ("presence", "laterality", "location", "severity"):
                op = {
                    "presence": "SET_NEGATION",
                    "laterality": "SET_LATERALITY",
                    "location": "SET_LOCATION",
                    "severity": "SET_SEVERITY",
                }[attribute]
                if (finding, op) in gold_targets or any(e.finding == finding and e.op in {"ADD_FINDING", "REMOVE_FINDING"} for e in gold.values()):
                    continue
                preserve_total += 1
                preserved += int(getattr(pred_slot, attribute) == getattr(corr_slot, attribute))

    n = max(1, len(rows))
    result = {
        "n": len(rows),
        "operator_f0.5": _f1(tp, fp, fn, beta),
        "operator_f1": _f1(tp, fp, fn, 1.0),
        "exact_composition_fix_rate": exact / n,
        "non_target_preservation": preserved / max(1, preserve_total),
        "clean_false_edit_rate": clean_false_edits / max(1, clean),
        "by_k": {str(k): {"n": v["n"], "exact": v["exact"] / max(1, v["n"])} for k, v in sorted(by_k.items())},
        "by_operator": {
            op: {"f0.5": _f1(v["tp"], v["fp"], v["fn"], beta), "f1": _f1(v["tp"], v["fp"], v["fn"], 1.0)}
            for op, v in sorted(by_operator.items())
        },
    }
    return result


def composition_gap(seen: Mapping[str, object], unseen: Mapping[str, object]) -> float:
    return float(seen.get("exact_composition_fix_rate", 0.0)) - float(unseen.get("exact_composition_fix_rate", 0.0))
