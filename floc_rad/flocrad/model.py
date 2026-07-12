from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

from .schema import Edit, encode_edits


class SimpleTokenizer:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, vocab: Optional[Mapping[str, int]] = None, max_length: int = 192):
        self.vocab = dict(vocab or {self.PAD: 0, self.UNK: 1})
        self.max_length = max_length

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[^\w\s]", (text or "").lower())

    def fit(self, texts: Iterable[str], min_frequency: int = 2, max_vocab: int = 30000) -> None:
        counts = Counter(token for text in texts for token in self.tokenize(text))
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        self.vocab = {self.PAD: 0, self.UNK: 1}
        for token, count in ordered:
            if count < min_frequency or len(self.vocab) >= max_vocab:
                break
            self.vocab[token] = len(self.vocab)

    def encode(self, text: str) -> Tuple[List[int], int]:
        tokens = self.tokenize(text)[: self.max_length]
        ids = [self.vocab.get(token, 1) for token in tokens]
        length = max(1, len(ids))
        ids = ids + [0] * (self.max_length - len(ids))
        return ids, length

    def save(self, path: str) -> None:
        Path(path).write_text(
            json.dumps({"vocab": self.vocab, "max_length": self.max_length}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["vocab"], int(data["max_length"]))


def load_radiograph(path: str) -> Image.Image:
    image_path = Path(path)
    if image_path.suffix.lower() == ".dcm":
        try:
            import pydicom
        except ImportError as exc:
            raise ImportError("DICOM input requires pydicom. Install floc_rad/requirements.txt.") from exc
        ds = pydicom.dcmread(str(image_path))
        pixels = ds.pixel_array.astype(np.float32)
        pixels -= pixels.min()
        pixels /= max(float(pixels.max()), 1e-6)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = 1.0 - pixels
        array = (pixels * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    return Image.open(image_path).convert("RGB")


class CorrectionDataset(Dataset):
    def __init__(
        self,
        items: Sequence[Mapping[str, object]],
        tokenizer: SimpleTokenizer,
        vocabulary: Sequence[Edit],
        image_size: int = 224,
        augment: bool = False,
    ):
        self.items = list(items)
        self.tokenizer = tokenizer
        self.vocabulary = list(vocabulary)
        operations = [transforms.Resize((image_size, image_size))]
        if augment:
            operations.extend([
                transforms.RandomAffine(degrees=3, translate=(0.02, 0.02)),
                transforms.RandomAutocontrast(p=0.2),
            ])
        operations.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        ])
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = self.items[index]
        image = self.transform(load_radiograph(str(item["image_path"])))
        input_ids, length = self.tokenizer.encode(str(item["corrupted_report"]))
        edits = [Edit.from_dict(edit) for edit in item.get("edits", [])]
        labels = torch.tensor(encode_edits(edits, self.vocabulary), dtype=torch.float32)
        return {
            "image": image,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "labels": labels,
            "id": str(item["id"]),
            "composition": str(item["composition"]),
            "k": int(item["k"]),
            "raw": item,
        }


def collate_batch(batch: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "length": torch.stack([item["length"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "id": [item["id"] for item in batch],
        "composition": [item["composition"] for item in batch],
        "k": [item["k"] for item in batch],
        "raw": [item["raw"] for item in batch],
    }


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 192, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.detach().cpu().clamp_min(1),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        features = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.dropout(self.norm(features))


class ImageEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, freeze: bool = False):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT if backbone == "resnet18" else models.ResNet50_Weights.DEFAULT
            except Exception:
                weights = None
        if backbone == "tiny":
            network = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            output_dim = 64
        elif backbone == "resnet18":
            network = models.resnet18(weights=weights)
            output_dim = network.fc.in_features
            network.fc = nn.Identity()
        elif backbone == "resnet50":
            network = models.resnet50(weights=weights)
            output_dim = network.fc.in_features
            network.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported image backbone: {backbone}")
        self.network = network
        self.output_dim = output_dim
        if freeze:
            for parameter in self.network.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class MultimodalEditPredictor(nn.Module):
    """A single predictor for factorized clinical edits."""

    def __init__(
        self,
        vocab_size: int,
        num_edits: int,
        image_backbone: str = "resnet18",
        image_pretrained: bool = True,
        freeze_image: bool = False,
        text_embed_dim: int = 192,
        text_hidden_dim: int = 256,
        fusion_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(image_backbone, image_pretrained, freeze_image)
        self.text_encoder = TextEncoder(vocab_size, text_embed_dim, text_hidden_dim, dropout)
        input_dim = self.image_encoder.output_dim + text_hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.edit_head = nn.Linear(fusion_dim, num_edits)

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(input_ids, length)
        fused = self.fusion(torch.cat([image_features, text_features], dim=-1))
        return self.edit_head(fused)


def save_checkpoint(
    path: str,
    model: MultimodalEditPredictor,
    tokenizer: SimpleTokenizer,
    vocabulary: Sequence[Edit],
    model_config: Mapping[str, object],
    threshold: float,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_config": dict(model_config),
        "vocabulary": [edit.to_dict() for edit in vocabulary],
        "tokenizer": {"vocab": tokenizer.vocab, "max_length": tokenizer.max_length},
        "threshold": threshold,
    }, output)


def load_checkpoint(path: str, device: str = "cpu") -> Tuple[MultimodalEditPredictor, SimpleTokenizer, List[Edit], float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    tokenizer = SimpleTokenizer(checkpoint["tokenizer"]["vocab"], checkpoint["tokenizer"]["max_length"])
    vocabulary = [Edit.from_dict(item) for item in checkpoint["vocabulary"]]
    model_config = checkpoint["model_config"]
    model = MultimodalEditPredictor(
        vocab_size=len(tokenizer.vocab),
        num_edits=len(vocabulary),
        **model_config,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer, vocabulary, float(checkpoint.get("threshold", 0.5))
