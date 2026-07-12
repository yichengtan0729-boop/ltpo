from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

FINDINGS: Tuple[str, ...] = (
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "lung_opacity",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
)

FINDING_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "atelectasis": ("atelectasis", "atelectatic"),
    "cardiomegaly": ("cardiomegaly", "enlarged cardiac silhouette", "enlarged heart", "cardiac enlargement"),
    "consolidation": ("consolidation", "consolidative opacity", "airspace consolidation"),
    "edema": ("pulmonary edema", "interstitial edema", "edema"),
    "lung_opacity": ("lung opacity", "pulmonary opacity", "airspace opacity", "focal opacity", "opacity"),
    "pleural_effusion": ("pleural effusion", "effusion"),
    "pneumonia": ("pneumonia", "infectious infiltrate"),
    "pneumothorax": ("pneumothorax", "ptx"),
}

NEGATION_CUES = (
    "no ", "without ", "negative for ", "absence of ", "free of ", "no evidence of ",
)
UNCERTAINTY_CUES = (
    "possible", "possibly", "may represent", "may be", "cannot exclude", "questionable", "suggestive of",
)
LATERALITY_ALIASES = {
    "left": ("left", "left-sided", "left sided"),
    "right": ("right", "right-sided", "right sided"),
    "bilateral": ("bilateral", "both lungs", "both sides", "biapical"),
}
LOCATION_ALIASES = {
    "upper": ("upper lobe", "upper lung", "apical", "apex"),
    "middle": ("middle lobe", "mid lung", "midlung"),
    "lower": ("lower lobe", "lower lung", "basilar", "basal", "base"),
    "perihilar": ("perihilar", "hilar"),
    "diffuse": ("diffuse", "throughout", "multifocal"),
}
SEVERITY_ALIASES = {
    "mild": ("mild", "small", "trace", "minimal", "slight"),
    "moderate": ("moderate", "medium"),
    "severe": ("severe", "large", "marked", "extensive"),
}

OP_ORDER = {
    "ADD_FINDING": 0,
    "REMOVE_FINDING": 0,
    "SET_NEGATION": 1,
    "SET_LATERALITY": 2,
    "SET_LOCATION": 3,
    "SET_SEVERITY": 4,
}


@dataclass
class FindingState:
    presence: str = "unmentioned"
    laterality: str = "unknown"
    location: str = "unknown"
    severity: str = "unknown"

    def to_dict(self) -> Dict[str, str]:
        return {
            "presence": self.presence,
            "laterality": self.laterality,
            "location": self.location,
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, str]) -> "FindingState":
        return cls(
            presence=str(data.get("presence", "unmentioned")),
            laterality=str(data.get("laterality", "unknown")),
            location=str(data.get("location", "unknown")),
            severity=str(data.get("severity", "unknown")),
        )


ClinicalState = Dict[str, FindingState]


@dataclass(frozen=True)
class Edit:
    op: str
    finding: str
    value: str = ""

    def __post_init__(self) -> None:
        if self.op not in OP_ORDER:
            raise ValueError(f"Unsupported operator: {self.op}")
        if self.finding not in FINDINGS:
            raise ValueError(f"Unsupported finding: {self.finding}")

    @property
    def key(self) -> str:
        return f"{self.op}|{self.finding}|{self.value}"

    @property
    def type_key(self) -> str:
        return self.op

    @property
    def finding_key(self) -> str:
        return f"{self.op}|{self.finding}"

    def to_dict(self) -> Dict[str, str]:
        return {"op": self.op, "finding": self.finding, "value": self.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, str]) -> "Edit":
        return cls(op=str(data["op"]), finding=str(data["finding"]), value=str(data.get("value", "")))


def empty_state() -> ClinicalState:
    return {finding: FindingState() for finding in FINDINGS}


def clone_state(state: Mapping[str, FindingState]) -> ClinicalState:
    return {finding: FindingState.from_dict(value.to_dict()) for finding, value in state.items()}


def state_to_json(state: Mapping[str, FindingState]) -> str:
    return json.dumps({key: value.to_dict() for key, value in state.items()}, sort_keys=True)


def state_from_json(text: str) -> ClinicalState:
    raw = json.loads(text)
    state = empty_state()
    for finding, attrs in raw.items():
        if finding in state:
            state[finding] = FindingState.from_dict(attrs)
    return state


def _extract_categorical(window: str, aliases: Mapping[str, Sequence[str]], default: str = "unknown") -> str:
    lowered = window.lower()
    for label, values in aliases.items():
        for value in values:
            if re.search(rf"\b{re.escape(value)}\b", lowered):
                return label
    return default


def parse_report(report: str) -> ClinicalState:
    """Parse a report into a compact, deterministic clinical state.

    Multiple mentions are aggregated with the same priority used by common
    chest-radiograph labelers: present > uncertain > absent > unmentioned.
    Attributes are taken from the highest-priority mention that states them.
    """
    text = re.sub(r"\s+", " ", report or "").strip()
    state = empty_state()
    if not text:
        return state

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    priority = {"unmentioned": 0, "absent": 1, "uncertain": 2, "present": 3}

    for finding, aliases in FINDING_ALIASES.items():
        best = FindingState()
        for sentence in sentences:
            lowered = sentence.lower()
            matches = []
            for alias in aliases:
                matches.extend(re.finditer(rf"\b{re.escape(alias)}\b", lowered))
            if not matches:
                continue
            mention = min(matches, key=lambda match: match.start())
            prefix = lowered[max(0, mention.start() - 55):mention.start()]
            is_negated = any(cue in prefix for cue in NEGATION_CUES)
            is_uncertain = any(cue in lowered for cue in UNCERTAINTY_CUES)
            presence = "absent" if is_negated else "uncertain" if is_uncertain else "present"
            candidate = FindingState(
                presence=presence,
                laterality=_extract_categorical(sentence, LATERALITY_ALIASES),
                location=_extract_categorical(sentence, LOCATION_ALIASES),
                severity=_extract_categorical(sentence, SEVERITY_ALIASES),
            )
            if priority[candidate.presence] > priority[best.presence]:
                best = candidate
            elif priority[candidate.presence] == priority[best.presence]:
                for attribute in ("laterality", "location", "severity"):
                    if getattr(best, attribute) == "unknown" and getattr(candidate, attribute) != "unknown":
                        setattr(best, attribute, getattr(candidate, attribute))
        state[finding] = best
    return state


def canonical_edits(source: Mapping[str, FindingState], target: Mapping[str, FindingState]) -> List[Edit]:
    """Return a unique minimal edit list from source to target."""
    edits: List[Edit] = []
    for finding in FINDINGS:
        src = source.get(finding, FindingState())
        dst = target.get(finding, FindingState())

        if src.presence != dst.presence:
            if dst.presence == "unmentioned":
                edits.append(Edit("REMOVE_FINDING", finding))
                continue
            if src.presence == "unmentioned":
                edits.append(Edit("ADD_FINDING", finding, dst.presence))
            else:
                edits.append(Edit("SET_NEGATION", finding, dst.presence))

        if dst.presence in {"present", "uncertain"}:
            if src.laterality != dst.laterality and dst.laterality != "unknown":
                edits.append(Edit("SET_LATERALITY", finding, dst.laterality))
            if src.location != dst.location and dst.location != "unknown":
                edits.append(Edit("SET_LOCATION", finding, dst.location))
            if src.severity != dst.severity and dst.severity != "unknown":
                edits.append(Edit("SET_SEVERITY", finding, dst.severity))

    return sorted(edits, key=lambda edit: (OP_ORDER[edit.op], edit.finding, edit.value))


def apply_edits(state: Mapping[str, FindingState], edits: Iterable[Edit]) -> ClinicalState:
    result = clone_state(state)
    for edit in sorted(edits, key=lambda item: (OP_ORDER[item.op], item.finding, item.value)):
        slot = result[edit.finding]
        if edit.op == "ADD_FINDING":
            slot.presence = edit.value or "present"
        elif edit.op == "REMOVE_FINDING":
            result[edit.finding] = FindingState()
        elif edit.op == "SET_NEGATION":
            slot.presence = edit.value
            if edit.value == "unmentioned":
                result[edit.finding] = FindingState()
        elif edit.op == "SET_LATERALITY":
            if slot.presence == "unmentioned":
                slot.presence = "present"
            slot.laterality = edit.value
        elif edit.op == "SET_LOCATION":
            if slot.presence == "unmentioned":
                slot.presence = "present"
            slot.location = edit.value
        elif edit.op == "SET_SEVERITY":
            if slot.presence == "unmentioned":
                slot.presence = "present"
            slot.severity = edit.value
    return result


def _finding_phrase(finding: str) -> str:
    return finding.replace("_", " ")


def realize_state(state: Mapping[str, FindingState], include_negative: bool = True) -> str:
    """Render a deterministic report used by the controlled benchmark."""
    positive: List[str] = []
    negative: List[str] = []
    uncertain: List[str] = []
    for finding in FINDINGS:
        slot = state.get(finding, FindingState())
        phrase = _finding_phrase(finding)
        if slot.presence == "unmentioned":
            continue
        if slot.presence == "absent":
            if include_negative:
                negative.append(f"No {phrase}.")
            continue

        modifiers: List[str] = []
        if slot.severity != "unknown":
            modifiers.append(slot.severity)
        if slot.laterality != "unknown":
            modifiers.append(slot.laterality)
        if slot.location != "unknown":
            modifiers.append(f"{slot.location} lung")
        description = " ".join(modifiers + [phrase]).strip()
        sentence = f"There is {description}."
        if slot.presence == "uncertain":
            sentence = f"Possible {description}."
            uncertain.append(sentence)
        else:
            positive.append(sentence)

    sentences = positive + uncertain + negative
    return " ".join(sentences) if sentences else "No acute cardiopulmonary abnormality."


def edit_vocabulary(findings: Sequence[str] = FINDINGS) -> List[Edit]:
    vocab: List[Edit] = []
    for finding in findings:
        vocab.extend([
            Edit("ADD_FINDING", finding, "present"),
            Edit("ADD_FINDING", finding, "uncertain"),
            Edit("REMOVE_FINDING", finding),
            Edit("SET_NEGATION", finding, "present"),
            Edit("SET_NEGATION", finding, "absent"),
            Edit("SET_NEGATION", finding, "uncertain"),
        ])
        for value in ("left", "right", "bilateral"):
            vocab.append(Edit("SET_LATERALITY", finding, value))
        for value in ("upper", "middle", "lower", "perihilar", "diffuse"):
            vocab.append(Edit("SET_LOCATION", finding, value))
        for value in ("mild", "moderate", "severe"):
            vocab.append(Edit("SET_SEVERITY", finding, value))
    return vocab


def composition_id(edits: Sequence[Edit], unit: str = "operator_finding") -> str:
    if unit == "operator":
        tokens = [edit.type_key for edit in edits]
    elif unit == "operator_finding":
        tokens = [edit.finding_key for edit in edits]
    elif unit == "full":
        tokens = [edit.key for edit in edits]
    else:
        raise ValueError(f"Unknown composition unit: {unit}")
    return "+".join(sorted(tokens)) if tokens else "CLEAN"


def encode_edits(edits: Sequence[Edit], vocabulary: Sequence[Edit]) -> List[int]:
    keys = {edit.key for edit in edits}
    return [1 if item.key in keys else 0 for item in vocabulary]


def decode_edits(scores: Sequence[float], vocabulary: Sequence[Edit], threshold: float = 0.5) -> List[Edit]:
    return [edit for edit, score in zip(vocabulary, scores) if float(score) >= threshold]
