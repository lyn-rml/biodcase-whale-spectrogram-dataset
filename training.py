#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError("PyTorch is not installed in this environment.") from exc

# Hardcoded execution constants
PREPROC_DIR = Path("preprocessed")
MODEL_PATH = Path("preprocessed/mlp_whale_classifier.pt")
BATCH_SIZE = 128
EPOCHS = 12
LEARNING_RATE = 1e-3
HIDDEN_1 = 256
HIDDEN_2 = 128
SEED = 42


class SpectrogramDataset(Dataset):
    def __init__(self, x_memmap: np.ndarray, labels: np.ndarray, frequencies: np.ndarray | None = None):
        self.x_memmap = x_memmap
        self.labels = labels.astype(np.int64, copy=False)
        self.frequencies = frequencies

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = np.asarray(self.x_memmap[idx], dtype=np.float32).reshape(-1)
        y_val = self.labels[idx]
        
        if self.frequencies is not None:
            freq_features = self.frequencies[idx].astype(np.float32)
            return torch.from_numpy(x), torch.from_numpy(freq_features), torch.tensor(y_val, dtype=torch.long)
        else:
            return torch.from_numpy(x), torch.tensor(y_val, dtype=torch.long)


class MLPClassifier(nn.Module):
    def __init__(self, in_features: int, out_classes: int, hidden_1: int = 256, hidden_2: int = 128, use_frequencies: bool = False):
        super().__init__()
        self.use_frequencies = use_frequencies
        
        # Input dimension includes frequency features if enabled
        total_input_features = in_features + (2 if use_frequencies else 0)
        
        self.net = nn.Sequential(
            nn.Linear(total_input_features, hidden_1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_2, out_classes),
        )

    def forward(self, x, freq=None):
        if self.use_frequencies and freq is not None:
            x = torch.cat([x, freq], dim=1)
        return self.net(x)


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    class_counts = np.bincount(y.astype(np.int64), minlength=num_classes)
    if np.any(class_counts == 0):
        missing = [int(i) for i, count in enumerate(class_counts) if count == 0]
        raise ValueError(f"Cannot compute class weights, zero samples for class ids: {missing}")
    return class_counts.sum() / (num_classes * class_counts.astype(np.float64))


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x_path = PREPROC_DIR / "X_normalized.npy"
    y_path = PREPROC_DIR / "y_labels.npy"
    label_names_path = PREPROC_DIR / "label_names.npy"
    frequencies_path = PREPROC_DIR / "frequencies.npy"

    for path in [x_path, y_path, label_names_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path)
    label_names = np.load(label_names_path)
    
    # Load frequencies if available
    frequencies = None
    use_frequencies = False
    if frequencies_path.exists():
        frequencies = np.load(frequencies_path)
        use_frequencies = True
        print(f"Loaded frequency features from {frequencies_path}")
    else:
        print(f"Frequency features not found at {frequencies_path} - training without frequency information")

    input_shape = X.shape[1:]
    input_dim = int(np.prod(input_shape))
    num_classes = int(len(label_names))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Samples: {len(y)}")
    print(f"Input shape: {input_shape} -> flattened dim: {input_dim}")
    print(f"Classes ({num_classes}): {list(label_names)}")
    if use_frequencies:
        print(f"Frequency features shape: {frequencies.shape}")

    dataset = SpectrogramDataset(X, y, frequencies)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = MLPClassifier(
        in_features=input_dim,
        out_classes=num_classes,
        hidden_1=HIDDEN_1,
        hidden_2=HIDDEN_2,
        use_frequencies=use_frequencies,
    ).to(device)

    class_weights_np = compute_class_weights(y, num_classes)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    class_counts = np.bincount(y.astype(np.int64), minlength=num_classes)
    print("Class weighting (count -> weight):")
    for i, name in enumerate(label_names):
        print(f"  {name}: {int(class_counts[i])} -> {class_weights_np[i]:.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_data in loader:
            if use_frequencies:
                xb, freq_b, yb = batch_data
                freq_b = freq_b.to(device, non_blocking=True)
            else:
                xb, yb = batch_data
                freq_b = None
            
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb, freq_b) if use_frequencies else model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * yb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch:02d}/{EPOCHS} | loss={epoch_loss:.4f} | acc={epoch_acc:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_shape": tuple(int(v) for v in input_shape),
            "label_names": [str(v) for v in label_names],
            "use_frequencies": bool(use_frequencies),
            "config": {
                "hidden_1": int(HIDDEN_1),
                "hidden_2": int(HIDDEN_2),
                "batch_size": int(BATCH_SIZE),
                "epochs": int(EPOCHS),
                "learning_rate": float(LEARNING_RATE),
                "seed": int(SEED),
            },
        },
        MODEL_PATH,
    )

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
