from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .schema import (
    FINDINGS,
    Edit,
    FindingState,
    apply_edits,
    canonical_edits,
    composition_id,
    empty_state,
    parse_report,
    realize_state,
    state_to_json,
)

OPENI_IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
OPENI_REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
MIMIC_CXR_BASE = "https://physionet.org/files/mimic-cxr/2.1.0"
MIMIC_JPG_BASE = "https://physionet.org/files/mimic-cxr-jpg/2.1.0"
CHEXPERT_PLUS_URL = "https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x?v=next"


@dataclass
class StudyRecord:
    study_id: str
    patient_id: str
    image_path: str
    report: str
    split: str = "train"
    dataset: str = "unknown"

    def to_dict(self) -> Dict[str, str]:
        return {
            "study_id": self.study_id,
            "patient_id": self.patient_id,
            "image_path": self.image_path,
            "report": self.report,
            "split": self.split,
            "dataset": self.dataset,
        }


def _run(
    command: Sequence[str],
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[str] = None,
) -> None:
    print("+", " ".join(command))
    subprocess.run(
        list(command),
        check=True,
        env=dict(os.environ, **(dict(env or {}))),
        cwd=cwd,
    )


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    print(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, destination)
    return destination


def _extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)
    elif archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(destination)
    else:
        raise ValueError(f"Unsupported archive: {archive}")


def download_openi(output_dir: str) -> None:
    root = Path(output_dir).expanduser().resolve()
    downloads = root / "downloads"
    images_archive = _download(OPENI_IMAGES_URL, downloads / "NLMCXR_png.tgz")
    reports_archive = _download(OPENI_REPORTS_URL, downloads / "NLMCXR_reports.tgz")
    _extract(images_archive, root)
    _extract(reports_archive, root)
    print(f"OpenI extracted under {root}")


def download_mimic(
    output_dir: str,
    username: str,
    password: str,
    max_images: int = 0,
) -> None:
    """Download MIMIC reports, metadata, and an optional image subset."""
    if not username or not password:
        raise ValueError("MIMIC download requires PHYSIONET_USERNAME and PHYSIONET_PASSWORD.")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    auth = ["--user", username, "--password", password]

    report_zip = root / "mimic-cxr-reports.zip"
    if not report_zip.exists():
        _run(["wget", "-c", *auth, f"{MIMIC_CXR_BASE}/mimic-cxr-reports.zip", "-O", str(report_zip)])
    reports_dir = root / "reports"
    if not reports_dir.exists():
        _extract(report_zip, reports_dir)

    filenames = [
        "mimic-cxr-2.0.0-split.csv.gz",
        "mimic-cxr-2.0.0-metadata.csv.gz",
        "mimic-cxr-2.0.0-chexpert.csv.gz",
        "RECORDS",
    ]
    for filename in filenames:
        path = root / filename
        if not path.exists():
            _run(["wget", "-c", *auth, f"{MIMIC_JPG_BASE}/{filename}", "-O", str(path)])

    records = root / "RECORDS"
    selected = root / "RECORDS.selected"
    lines = records.read_text(encoding="utf-8").splitlines()
    if max_images > 0:
        lines = lines[:max_images]
    selected.write_text("\n".join(lines) + "\n", encoding="utf-8")
    images_root = root / "jpg"
    images_root.mkdir(exist_ok=True)
    _run([
        "wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=1", *auth,
        "-i", str(selected), "--base", f"{MIMIC_JPG_BASE}/",
    ], cwd=str(images_root))
    print(f"MIMIC-CXR staged under {root}")


def stage_chexpert_plus(source_dir: str, output_dir: str, symlink: bool = True) -> None:
    source = Path(source_dir).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(
            f"CheXpert Plus source directory does not exist: {source}. "
            f"Download it after accepting the terms at {CHEXPERT_PLUS_URL}."
        )
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"CheXpert Plus target already exists: {target}")
        return
    if symlink:
        target.symlink_to(source, target_is_directory=True)
    else:
        shutil.copytree(source, target)
    print(f"CheXpert Plus staged at {target}")


def _extract_openi_text(root: ET.Element, tags: Sequence[str]) -> str:
    parts: List[str] = []
    wanted = {tag.lower() for tag in tags}
    for node in root.iter():
        if node.tag.lower().split("}")[-1] in wanted and node.text:
            parts.append(node.text.strip())
    return " ".join(part for part in parts if part)


def build_openi_manifest(data_dir: str, output_csv: str, seed: int = 42) -> pd.DataFrame:
    root = Path(data_dir).expanduser().resolve()
    xml_files = list(root.rglob("*.xml"))
    image_files = {path.stem: path for path in root.rglob("*.png")}
    records: List[StudyRecord] = []
    rng = random.Random(seed)

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        xml_root = tree.getroot()
        report = _extract_openi_text(xml_root, ("findings", "impression"))
        if not report:
            continue
        image_ids: List[str] = []
        for node in xml_root.iter():
            tag = node.tag.lower().split("}")[-1]
            if tag in {"parentimage", "image", "figure"}:
                candidate = node.attrib.get("id") or node.attrib.get("uid")
                if candidate:
                    image_ids.append(candidate)
            if tag in {"imageid", "id"} and node.text:
                image_ids.append(node.text.strip())
        image_path = next((image_files[item] for item in image_ids if item in image_files), None)
        if image_path is None:
            image_path = image_files.get(xml_path.stem)
        if image_path is None:
            continue
        study_id = xml_path.stem
        records.append(StudyRecord(study_id, study_id, str(image_path), report, dataset="openi"))

    rng.shuffle(records)
    n = len(records)
    for idx, record in enumerate(records):
        record.split = "train" if idx < int(0.8 * n) else "validate" if idx < int(0.9 * n) else "test"
    frame = pd.DataFrame([record.to_dict() for record in records])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def _read_report_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    sections = re.split(r"\n\s*([A-Z][A-Z /_-]{2,}):\s*", text)
    if len(sections) > 1:
        section_map = {sections[i].strip().lower(): sections[i + 1].strip() for i in range(1, len(sections) - 1, 2)}
        report = " ".join(section_map.get(name, "") for name in ("findings", "impression"))
        if report.strip():
            return re.sub(r"\s+", " ", report).strip()
    return re.sub(r"\s+", " ", text).strip()


def build_mimic_manifest(data_dir: str, output_csv: str) -> pd.DataFrame:
    root = Path(data_dir).expanduser().resolve()
    split_candidates = list(root.rglob("mimic-cxr-2.0.0-split.csv.gz"))
    metadata_candidates = list(root.rglob("mimic-cxr-2.0.0-metadata.csv.gz"))
    if not split_candidates:
        raise FileNotFoundError("Could not find mimic-cxr-2.0.0-split.csv.gz")
    split_df = pd.read_csv(split_candidates[0])
    metadata_df = pd.read_csv(metadata_candidates[0]) if metadata_candidates else pd.DataFrame()
    if not metadata_df.empty and "ViewPosition" in metadata_df:
        preferred = metadata_df[metadata_df["ViewPosition"].isin(["PA", "AP"])]
        if not preferred.empty:
            split_df = split_df.merge(preferred[["dicom_id", "ViewPosition"]], on="dicom_id", how="inner")
    split_df = split_df.drop_duplicates("study_id", keep="first")

    reports = {}
    for path in root.rglob("s*.txt"):
        match = re.match(r"s(\d+)\.txt$", path.name)
        if match:
            reports[int(match.group(1))] = path
    images = {path.stem: path for path in root.rglob("*.jpg")}

    records: List[StudyRecord] = []
    for row in split_df.itertuples(index=False):
        report_path = reports.get(int(row.study_id))
        image_path = images.get(str(row.dicom_id))
        if report_path is None or image_path is None:
            continue
        report = _read_report_file(report_path)
        if not report:
            continue
        split = "validate" if str(row.split) in {"validate", "val"} else str(row.split)
        records.append(StudyRecord(str(row.study_id), str(row.subject_id), str(image_path), report, split, "mimic"))

    frame = pd.DataFrame([record.to_dict() for record in records])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def _guess_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for column in columns:
        lowered_column = column.lower()
        if any(candidate.lower() in lowered_column for candidate in candidates):
            return column
    return None


def build_chexpert_plus_manifest(data_dir: str, output_csv: str, seed: int = 42) -> pd.DataFrame:
    """Build a manifest from a Redivis export using schema heuristics."""
    root = Path(data_dir).expanduser().resolve()
    tables: List[pd.DataFrame] = []
    for path in list(root.rglob("*.csv")) + list(root.rglob("*.csv.gz")) + list(root.rglob("*.parquet")):
        try:
            frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        frame["__source_dir"] = str(path.parent)
        tables.append(frame)
    if not tables:
        raise FileNotFoundError("No CSV or Parquet metadata found in the CheXpert Plus export.")

    candidates: List[pd.DataFrame] = []
    for frame in tables:
        report_col = _guess_column(frame.columns, ("report", "report_text", "findings", "impression"))
        image_col = _guess_column(frame.columns, ("path", "image_path", "dicom_path", "file_path"))
        if report_col and image_col:
            out = pd.DataFrame({
                "report": frame[report_col].astype(str),
                "raw_image_path": frame[image_col].astype(str),
                "__source_dir": frame["__source_dir"],
            })
            study_col = _guess_column(frame.columns, ("study_id", "study_uid", "accession"))
            patient_col = _guess_column(frame.columns, ("patient_id", "subject_id"))
            out["study_id"] = frame[study_col].astype(str) if study_col else frame.index.astype(str)
            out["patient_id"] = frame[patient_col].astype(str) if patient_col else out["study_id"]
            candidates.append(out)
    if not candidates:
        raise ValueError(
            "Could not find a table containing both report text and an image path. "
            "Export the CheXpert Plus report/image mapping table from Redivis or provide a universal manifest."
        )
    frame = pd.concat(candidates, ignore_index=True).drop_duplicates("study_id")

    def resolve_path(row: pd.Series) -> str:
        raw = Path(str(row.raw_image_path))
        if raw.is_absolute() and raw.exists():
            return str(raw)
        candidate = Path(row.__source_dir) / raw
        if candidate.exists():
            return str(candidate.resolve())
        matches = list(root.rglob(raw.name))
        return str(matches[0]) if matches else str(candidate)

    frame["image_path"] = frame.apply(resolve_path, axis=1)
    frame = frame[frame["image_path"].map(lambda value: Path(value).exists())]
    rng = np.random.default_rng(seed)
    patient_ids = frame["patient_id"].drop_duplicates().to_numpy()
    rng.shuffle(patient_ids)
    train_cut, val_cut = int(0.8 * len(patient_ids)), int(0.9 * len(patient_ids))
    split_map = {pid: "train" for pid in patient_ids[:train_cut]}
    split_map.update({pid: "validate" for pid in patient_ids[train_cut:val_cut]})
    split_map.update({pid: "test" for pid in patient_ids[val_cut:]})
    frame["split"] = frame["patient_id"].map(split_map)
    frame["dataset"] = "chexpert_plus"
    frame = frame[["study_id", "patient_id", "image_path", "report", "split", "dataset"]]
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def build_universal_manifest(data_dir: str, output_csv: str, input_csv: str) -> pd.DataFrame:
    root = Path(data_dir).expanduser().resolve()
    frame = pd.read_csv(input_csv)
    required = {"study_id", "image_path", "report"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Universal manifest is missing columns: {sorted(missing)}")
    if "patient_id" not in frame:
        frame["patient_id"] = frame["study_id"]
    if "split" not in frame:
        frame["split"] = "train"
    if "dataset" not in frame:
        frame["dataset"] = "custom"
    frame["image_path"] = frame["image_path"].map(
        lambda value: str((root / str(value)).resolve()) if not Path(str(value)).is_absolute() else str(value)
    )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def generate_synthetic_manifest(output_dir: str, n: int = 120, seed: int = 42) -> str:
    root = Path(output_dir).expanduser().resolve()
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records: List[StudyRecord] = []

    for index in range(n):
        state = empty_state()
        active = rng.sample(list(FINDINGS), k=rng.randint(1, 3))
        for finding in active:
            state[finding] = FindingState(
                presence="present",
                laterality=rng.choice(["left", "right", "bilateral", "unknown"]),
                location=rng.choice(["upper", "middle", "lower", "perihilar", "diffuse", "unknown"]),
                severity=rng.choice(["mild", "moderate", "severe", "unknown"]),
            )
        for finding in rng.sample([item for item in FINDINGS if item not in active], k=1):
            state[finding] = FindingState(presence="absent")

        image = Image.new("L", (224, 224), color=20)
        draw = ImageDraw.Draw(image)
        for offset, finding in enumerate(active):
            slot = state[finding]
            x = 35 if slot.laterality == "left" else 135 if slot.laterality == "right" else 85
            y = 35 if slot.location == "upper" else 140 if slot.location == "lower" else 90
            radius = 8 if slot.severity == "mild" else 16 if slot.severity == "moderate" else 24
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=140 + 20 * offset)
        image_path = images_dir / f"study_{index:05d}.png"
        image.save(image_path)
        split = "train" if index < int(0.7 * n) else "validate" if index < int(0.85 * n) else "test"
        records.append(StudyRecord(str(index), str(index // 2), str(image_path), realize_state(state), split, "synthetic"))

    manifest = root / "manifest.csv"
    pd.DataFrame([record.to_dict() for record in records]).to_csv(manifest, index=False)
    return str(manifest)


def _possible_corruptions(gold: Mapping[str, FindingState]) -> List[Tuple[Edit, Edit]]:
    pairs: List[Tuple[Edit, Edit]] = []
    for finding in FINDINGS:
        slot = gold[finding]
        if slot.presence in {"present", "uncertain"}:
            pairs.append((Edit("REMOVE_FINDING", finding), Edit("ADD_FINDING", finding, slot.presence)))
            new_presence = "absent" if slot.presence != "absent" else "present"
            pairs.append((Edit("SET_NEGATION", finding, new_presence), Edit("SET_NEGATION", finding, slot.presence)))
            if slot.laterality != "unknown":
                alternatives = [value for value in ("left", "right", "bilateral") if value != slot.laterality]
                for value in alternatives:
                    pairs.append((Edit("SET_LATERALITY", finding, value), Edit("SET_LATERALITY", finding, slot.laterality)))
            if slot.location != "unknown":
                alternatives = [value for value in ("upper", "middle", "lower", "perihilar", "diffuse") if value != slot.location]
                for value in alternatives[:2]:
                    pairs.append((Edit("SET_LOCATION", finding, value), Edit("SET_LOCATION", finding, slot.location)))
            if slot.severity != "unknown":
                alternatives = [value for value in ("mild", "moderate", "severe") if value != slot.severity]
                for value in alternatives:
                    pairs.append((Edit("SET_SEVERITY", finding, value), Edit("SET_SEVERITY", finding, slot.severity)))
        elif slot.presence == "absent":
            pairs.append((Edit("SET_NEGATION", finding, "present"), Edit("SET_NEGATION", finding, "absent")))
        elif slot.presence == "unmentioned":
            pairs.append((Edit("ADD_FINDING", finding, "present"), Edit("REMOVE_FINDING", finding)))
    return pairs


def _select_independent_pairs(
    pairs: Sequence[Tuple[Edit, Edit]],
    k: int,
    rng: random.Random,
) -> Optional[List[Tuple[Edit, Edit]]]:
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    selected: List[Tuple[Edit, Edit]] = []
    touched: set[Tuple[str, str]] = set()
    for corruption, recovery in shuffled:
        attribute = {
            "ADD_FINDING": "presence",
            "REMOVE_FINDING": "presence",
            "SET_NEGATION": "presence",
            "SET_LATERALITY": "laterality",
            "SET_LOCATION": "location",
            "SET_SEVERITY": "severity",
        }[corruption.op]
        key = (corruption.finding, attribute)
        if key in touched:
            continue
        selected.append((corruption, recovery))
        touched.add(key)
        if len(selected) == k:
            return selected
    return None


def build_correction_dataset(
    manifest_csv: str,
    output_jsonl: str,
    examples_per_study: int = 3,
    error_counts: Sequence[int] = (1, 2, 3),
    clean_fraction: float = 0.1,
    composition_unit: str = "operator_finding",
    seed: int = 42,
    min_mentions: int = 1,
) -> Dict[str, int]:
    frame = pd.read_csv(manifest_csv)
    required = {"study_id", "patient_id", "image_path", "report", "split", "dataset"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    rng = random.Random(seed)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()

    with output_path.open("w", encoding="utf-8") as handle:
        for row in frame.itertuples(index=False):
            gold = parse_report(str(row.report))
            mentions = sum(slot.presence != "unmentioned" for slot in gold.values())
            if mentions < min_mentions:
                stats["skipped_unparsed"] += 1
                continue
            if rng.random() < clean_fraction:
                example = {
                    "id": f"{row.dataset}:{row.study_id}:clean",
                    "study_id": str(row.study_id),
                    "patient_id": str(row.patient_id),
                    "dataset": str(row.dataset),
                    "source_split": str(row.split),
                    "image_path": str(row.image_path),
                    "gold_report": str(row.report),
                    "corrupted_report": realize_state(gold),
                    "gold_state": state_to_json(gold),
                    "corrupted_state": state_to_json(gold),
                    "edits": [],
                    "composition": "CLEAN",
                    "k": 0,
                }
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
                stats["clean"] += 1

            pairs = _possible_corruptions(gold)
            for example_index in range(examples_per_study):
                k = int(rng.choice(list(error_counts)))
                selected = _select_independent_pairs(pairs, k, rng)
                if not selected:
                    stats["skipped_no_corruption"] += 1
                    continue
                corruptions = [item[0] for item in selected]
                corrupted = apply_edits(gold, corruptions)
                recoveries = canonical_edits(corrupted, gold)
                if not recoveries:
                    continue
                example = {
                    "id": f"{row.dataset}:{row.study_id}:{example_index}",
                    "study_id": str(row.study_id),
                    "patient_id": str(row.patient_id),
                    "dataset": str(row.dataset),
                    "source_split": str(row.split),
                    "image_path": str(row.image_path),
                    "gold_report": str(row.report),
                    "corrupted_report": realize_state(corrupted),
                    "gold_state": state_to_json(gold),
                    "corrupted_state": state_to_json(corrupted),
                    "edits": [edit.to_dict() for edit in recoveries],
                    "composition": composition_id(recoveries, composition_unit),
                    "k": len(recoveries),
                }
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
                stats[f"k{len(recoveries)}"] += 1
    return dict(stats)


def read_jsonl(path: str) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: Iterable[Mapping[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(dict(item), ensure_ascii=False) + "\n")


def make_composition_splits(
    input_jsonl: str,
    output_dir: str,
    holdout_fraction: float = 0.25,
    seed: int = 42,
    min_compound_count: int = 4,
) -> Dict[str, object]:
    items = read_jsonl(input_jsonl)
    train_candidates = [item for item in items if item["source_split"] == "train"]
    val_items = [item for item in items if item["source_split"] in {"validate", "val"}]
    test_candidates = [item for item in items if item["source_split"] == "test"]

    train_counts = Counter(str(item["composition"]) for item in train_candidates if item["composition"] != "CLEAN")
    test_counts = Counter(str(item["composition"]) for item in test_candidates if item["composition"] != "CLEAN")
    eligible = sorted(
        composition for composition, count in train_counts.items()
        if count >= min_compound_count and test_counts.get(composition, 0) >= 1
    )
    rng = random.Random(seed)
    rng.shuffle(eligible)
    n_holdout = max(1, int(round(len(eligible) * holdout_fraction))) if eligible else 0
    held_out = set(eligible[:n_holdout])

    train_items = [item for item in train_candidates if str(item["composition"]) not in held_out]
    val_items = [item for item in val_items if str(item["composition"]) not in held_out]
    seen_test = [item for item in test_candidates if item["composition"] == "CLEAN" or str(item["composition"]) not in held_out]
    unseen_test = [item for item in test_candidates if str(item["composition"]) in held_out]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(output / "train.jsonl"), train_items)
    write_jsonl(str(output / "val.jsonl"), val_items)
    write_jsonl(str(output / "test_seen.jsonl"), seen_test)
    write_jsonl(str(output / "test_unseen.jsonl"), unseen_test)
    metadata = {
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "held_out_compositions": sorted(held_out),
        "counts": {
            "train": len(train_items),
            "val": len(val_items),
            "test_seen": len(seen_test),
            "test_unseen": len(unseen_test),
        },
    }
    (output / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
