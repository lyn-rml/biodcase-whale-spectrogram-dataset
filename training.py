#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError("PyTorch is not installed in this environment.") from exc


class SpectrogramDataset(Dataset):
    def __init__(self, x_memmap: np.ndarray, labels: np.ndarray):
        self.x_memmap = x_memmap
        self.labels = labels.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = np.asarray(self.x_memmap[idx], dtype=np.float32).reshape(-1)
        y_val = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y_val, dtype=torch.long)


class MLPClassifier(nn.Module):
    def __init__(self, in_features: int, out_classes: int, hidden_1: int = 256, hidden_2: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_2, out_classes),
        )

    def forward(self, x):
        return self.net(x)


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    class_counts = np.bincount(y.astype(np.int64), minlength=num_classes)
    if np.any(class_counts == 0):
        missing = [int(i) for i, count in enumerate(class_counts) if count == 0]
        raise ValueError(f"Cannot compute class weights, zero samples for class ids: {missing}")
    return class_counts.sum() / (num_classes * class_counts.astype(np.float64))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP classifier with class-weighted cross-entropy.")
    parser.add_argument("--preproc-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--model-path", type=Path, default=Path("preprocessed/mlp_whale_classifier.pt"))

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-1", type=int, default=256)
    parser.add_argument("--hidden-2", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_path = args.preproc_dir / "X_normalized.npy"
    y_path = args.preproc_dir / "y_labels.npy"
    label_names_path = args.preproc_dir / "label_names.npy"

    for path in [x_path, y_path, label_names_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path)
    label_names = np.load(label_names_path)

    input_shape = X.shape[1:]
    input_dim = int(np.prod(input_shape))
    num_classes = int(len(label_names))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Samples: {len(y)}")
    print(f"Input shape: {input_shape} -> flattened dim: {input_dim}")
    print(f"Classes ({num_classes}): {list(label_names)}")

    dataset = SpectrogramDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = MLPClassifier(
        in_features=input_dim,
        out_classes=num_classes,
        hidden_1=args.hidden_1,
        hidden_2=args.hidden_2,
    ).to(device)

    class_weights_np = compute_class_weights(y, num_classes)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    class_counts = np.bincount(y.astype(np.int64), minlength=num_classes)
    print("Class weighting (count -> weight):")
    for i, name in enumerate(label_names):
        print(f"  {name}: {int(class_counts[i])} -> {class_weights_np[i]:.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * yb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch:02d}/{args.epochs} | loss={epoch_loss:.4f} | acc={epoch_acc:.4f}")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_shape": tuple(int(v) for v in input_shape),
            "label_names": [str(v) for v in label_names],
            "config": {
                "hidden_1": int(args.hidden_1),
                "hidden_2": int(args.hidden_2),
                "batch_size": int(args.batch_size),
                "epochs": int(args.epochs),
                "learning_rate": float(args.learning_rate),
                "seed": int(args.seed),
            },
        },
        args.model_path,
    )

    print(f"Model saved to: {args.model_path}")


if __name__ == "__main__":
    main()
