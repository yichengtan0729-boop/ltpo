from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import read_jsonl
from .metrics import composition_gap, evaluate_predictions
from .model import CorrectionDataset, MultimodalEditPredictor, SimpleTokenizer, collate_batch, load_checkpoint, save_checkpoint
from .schema import Edit, apply_edits, decode_edits, edit_vocabulary, realize_state, state_from_json


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_vocabulary(items: Sequence[Mapping[str, object]]) -> List[Edit]:
    observed = {}
    for item in items:
        for raw in item.get("edits", []):
            edit = Edit.from_dict(raw)
            observed[edit.key] = edit
    return sorted(observed.values(), key=lambda edit: (edit.op, edit.finding, edit.value))


def _pos_weight(items: Sequence[Mapping[str, object]], vocabulary: Sequence[Edit], cap: float = 30.0) -> torch.Tensor:
    counts = np.zeros(len(vocabulary), dtype=np.float64)
    index = {edit.key: idx for idx, edit in enumerate(vocabulary)}
    for item in items:
        for raw in item.get("edits", []):
            key = Edit.from_dict(raw).key
            if key in index:
                counts[index[key]] += 1
    negatives = max(1, len(items)) - counts
    weights = negatives / np.maximum(counts, 1.0)
    return torch.tensor(np.clip(weights, 1.0, cap), dtype=torch.float32)


def tune_threshold(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probabilities = torch.sigmoid(logits).cpu().numpy()
    gold = labels.cpu().numpy().astype(bool)
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.linspace(0.15, 0.85, 29):
        pred = probabilities >= threshold
        tp = np.logical_and(pred, gold).sum()
        fp = np.logical_and(pred, ~gold).sum()
        fn = np.logical_and(~pred, gold).sum()
        score = 0.0 if 1.25 * tp + 0.25 * fn + fp == 0 else 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    return best_threshold


def _run_epoch(model, loader, optimizer, device, pos_weight, grad_clip: float = 1.0) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        lengths = batch["length"].to(device)
        labels = batch["labels"].to(device)
        logits = model(images, input_ids, lengths)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += float(loss.item()) * labels.shape[0]
        total += labels.shape[0]
    return total_loss / max(1, total)


@torch.no_grad()
def collect_logits(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, labels = [], []
    for batch in loader:
        logits.append(model(batch["image"].to(device), batch["input_ids"].to(device), batch["length"].to(device)).cpu())
        labels.append(batch["labels"].cpu())
    if not logits:
        return torch.empty((0, model.edit_head.out_features)), torch.empty((0, model.edit_head.out_features))
    return torch.cat(logits), torch.cat(labels)


def train_model(
    train_jsonl: str,
    val_jsonl: str,
    output_dir: str,
    device: str = "cuda",
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    image_backbone: str = "resnet18",
    image_pretrained: bool = True,
    freeze_image: bool = False,
    image_size: int = 224,
    num_workers: int = 4,
    max_text_length: int = 192,
    patience: int = 3,
) -> str:
    seed_everything(seed)
    device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_items = read_jsonl(train_jsonl)
    val_items = read_jsonl(val_jsonl)
    if not train_items:
        raise ValueError("Training split is empty.")
    vocabulary = _build_vocabulary(train_items)
    if not vocabulary:
        raise ValueError("No edit labels were found in the training split.")
    tokenizer = SimpleTokenizer(max_length=max_text_length)
    tokenizer.fit([str(item["corrupted_report"]) for item in train_items], min_frequency=1)
    train_ds = CorrectionDataset(train_items, tokenizer, vocabulary, image_size, augment=True)
    val_ds = CorrectionDataset(val_items, tokenizer, vocabulary, image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_batch, pin_memory=device.startswith("cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch, pin_memory=device.startswith("cuda"))

    model_config = {
        "image_backbone": image_backbone,
        "image_pretrained": image_pretrained,
        "freeze_image": freeze_image,
    }
    model = MultimodalEditPredictor(len(tokenizer.vocab), len(vocabulary), **model_config).to(device)
    optimizer = AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=learning_rate, weight_decay=weight_decay)
    pos_weight = _pos_weight(train_items, vocabulary).to(device)
    best_loss = math.inf
    bad_epochs = 0
    checkpoint_path = output / "best.pt"
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, device, pos_weight)
        val_loss = _run_epoch(model, val_loader, None, device, pos_weight) if val_items else train_loss
        val_logits, val_labels = collect_logits(model, val_loader, device) if val_items else (torch.empty(0), torch.empty(0))
        threshold = tune_threshold(val_logits, val_labels) if val_logits.numel() else 0.5
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "threshold": threshold})
        print(f"epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f} threshold={threshold:.2f}")
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            bad_epochs = 0
            save_checkpoint(str(checkpoint_path), model, tokenizer, vocabulary, model_config, threshold)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return str(checkpoint_path)


@torch.no_grad()
def predict_items(
    checkpoint: str,
    items: Sequence[Mapping[str, object]],
    device: str = "cuda",
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 4,
    threshold: float | None = None,
) -> List[Dict[str, object]]:
    device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
    model, tokenizer, vocabulary, saved_threshold = load_checkpoint(checkpoint, device)
    threshold = saved_threshold if threshold is None else threshold
    dataset = CorrectionDataset(items, tokenizer, vocabulary, image_size, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
    rows: List[Dict[str, object]] = []
    model.eval()
    for batch in loader:
        probs = torch.sigmoid(model(batch["image"].to(device), batch["input_ids"].to(device), batch["length"].to(device))).cpu().numpy()
        for raw, scores in zip(batch["raw"], probs):
            pred_edits = decode_edits(scores, vocabulary, threshold)
            corrupted_state = state_from_json(str(raw["corrupted_state"]))
            corrected_state = apply_edits(corrupted_state, pred_edits)
            rows.append({
                "id": raw["id"],
                "composition": raw["composition"],
                "k": raw["k"],
                "corrupted_report": raw["corrupted_report"],
                "gold_report": raw["gold_report"],
                "corrected_report": realize_state(corrected_state),
                "corrupted_state": raw["corrupted_state"],
                "gold_state": raw["gold_state"],
                "gold_edits": raw.get("edits", []),
                "pred_edits": [edit.to_dict() for edit in pred_edits],
            })
    return rows


def evaluate_model(
    checkpoint: str,
    seen_jsonl: str,
    unseen_jsonl: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 4,
) -> Dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seen_rows = predict_items(checkpoint, read_jsonl(seen_jsonl), device, batch_size, image_size, num_workers)
    unseen_rows = predict_items(checkpoint, read_jsonl(unseen_jsonl), device, batch_size, image_size, num_workers)
    seen_metrics = evaluate_predictions(seen_rows)
    unseen_metrics = evaluate_predictions(unseen_rows)
    metrics = {"seen": seen_metrics, "unseen": unseen_metrics, "composition_gap": composition_gap(seen_metrics, unseen_metrics)}
    for name, rows in (("seen_predictions.jsonl", seen_rows), ("unseen_predictions.jsonl", unseen_rows)):
        with (output / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
