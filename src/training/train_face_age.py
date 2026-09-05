"""Train and evaluate a lightweight FG-NET subject-embedding baseline."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor

_IMAGE_PATTERN = re.compile(r"^(?P<subject>\d{3})A(?P<age>\d{2})", re.IGNORECASE)


def parse_fgnet_name(path: Path) -> tuple[str, int]:
    """Extract subject ID and age from an FG-NET filename."""
    match = _IMAGE_PATTERN.match(path.stem)
    if not match:
        raise ValueError(f"Unexpected FG-NET filename: {path.name}")
    return match.group("subject"), int(match.group("age"))


def split_fgnet(
    root: Path, *, test_fraction: float = 0.2
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold out the latest-age images within each subject for testing."""
    if not root.is_dir():
        raise FileNotFoundError(f"FG-NET root does not exist: {root}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.rglob("*.JPG")) + sorted(root.rglob("*.jpg")):
        subject, age = parse_fgnet_name(path)
        grouped[subject].append({"path": str(path), "subject": subject, "age": age})
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for subject, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (row["age"], row["path"]))
        test_count = max(1, round(len(rows) * test_fraction))
        train_rows.extend(rows[:-test_count])
        test_rows.extend(rows[-test_count:])
    if len(grouped) < 2 or not train_rows or not test_rows:
        raise ValueError("FG-NET needs at least two subjects with train and test images")
    return train_rows, test_rows


class FGNetDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], labels: dict[str, int]) -> None:
        self.rows = rows
        self.labels = labels
        self.transform = Compose([Resize((112, 112)), ToTensor()])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.labels[row["subject"]]


class EmbeddingNet(nn.Module):
    """Small CPU-friendly embedding model for a baseline experiment."""

    def __init__(self, classes: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Linear(64, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, classes)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(images).flatten(1)
        embeddings = nn.functional.normalize(self.embedding(features), dim=1)
        return self.classifier(embeddings), embeddings


def _embeddings(
    model: EmbeddingNet, rows: list[dict[str, Any]], labels: dict[str, int]
) -> tuple[np.ndarray, list[str]]:
    model.eval()
    loader = DataLoader(FGNetDataset(rows, labels), batch_size=32)
    vectors: list[np.ndarray] = []
    subjects: list[str] = []
    with torch.no_grad():
        for index, (images, _) in enumerate(loader):
            _, batch_embeddings = model(images)
            vectors.append(batch_embeddings.numpy())
            subjects.extend(row["subject"] for row in rows[index * 32 : index * 32 + len(images)])
    return np.concatenate(vectors), subjects


def evaluate_verification(
    model: EmbeddingNet, rows: list[dict[str, Any]], labels: dict[str, int]
) -> dict[str, float]:
    vectors, subjects = _embeddings(model, rows, labels)
    scores: list[float] = []
    targets: list[int] = []
    by_subject: dict[str, list[int]] = defaultdict(list)
    for index, subject in enumerate(subjects):
        by_subject[subject].append(index)
    subject_ids = sorted(by_subject)
    for subject in subject_ids:
        indices = by_subject[subject]
        for left, right in zip(indices, indices[1:]):
            scores.append(float(np.dot(vectors[left], vectors[right])))
            targets.append(1)
        other = subject_ids[(subject_ids.index(subject) + 1) % len(subject_ids)]
        for left, right in zip(indices, by_subject[other]):
            scores.append(float(np.dot(vectors[left], vectors[right])))
            targets.append(0)
    return {"roc_auc": round(float(roc_auc_score(targets, scores)), 4), "pairs": float(len(scores))}


def train(root: Path, output: Path, *, epochs: int = 5, batch_size: int = 32) -> dict[str, float]:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    train_rows, test_rows = split_fgnet(root)
    subjects = sorted({row["subject"] for row in train_rows})
    labels = {subject: index for index, subject in enumerate(subjects)}
    model = EmbeddingNet(len(labels))
    loader = DataLoader(FGNetDataset(train_rows, labels), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for images, targets in loader:
            optimizer.zero_grad()
            loss, _ = model(images)
            loss_function(loss, targets).backward()
            optimizer.step()
    metrics = evaluate_verification(model, test_rows, labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "labels": labels, "metrics": metrics}, output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("weights/fgnet-baseline.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(train(args.root, args.output, epochs=args.epochs, batch_size=args.batch_size)))


if __name__ == "__main__":
    main()