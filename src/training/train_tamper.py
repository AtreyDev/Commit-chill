"""Train an EfficientNet-B0 IDNet baseline from a prepared CSV manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b0
from torchvision.transforms import Compose, Resize, ToTensor


class ManifestImageDataset(Dataset):
    """Load authentic/tampered samples from a manifest without copying images."""

    def __init__(self, manifest: Path, split: str) -> None:
        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row["split"] == split]
        self.rows = [row for row in rows if row["label"] in {"authentic", "tampered"}]
        self.transform = Compose([Resize((224, 224)), ToTensor()])
        if not self.rows:
            raise ValueError(f"No authentic/tampered rows found for split: {split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"] == "tampered")


def train(
    manifest: Path,
    output: Path,
    *,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device: str | None = None,
) -> dict[str, float]:
    """Train and save a small baseline; intended for a prepared IDNet subset."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    train_data = ManifestImageDataset(manifest, "train")
    validation_data = ManifestImageDataset(manifest, "validation")
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(images.to(selected_device)), labels.to(selected_device))
            loss.backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in validation_loader:
            predictions = model(images.to(selected_device)).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
            total += labels.numel()
    accuracy = correct / total if total else 0.0
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "accuracy": accuracy}, output)
    return {"validation_accuracy": round(accuracy, 4), "samples": float(total)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("weights/idnet_efficientnet_b0.pt"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(train(args.manifest, args.output, epochs=args.epochs, batch_size=args.batch_size))


if __name__ == "__main__":
    main()