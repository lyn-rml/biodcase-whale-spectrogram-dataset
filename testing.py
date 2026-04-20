#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:
    raise ImportError("PyTorch is not installed in this environment.") from exc

from preprocessing_wavfiles import preprocess_split
from training import MLPClassifier, SpectrogramDataset

# Hardcoded execution constants
DATASET_ROOT = Path("biodcase_development_set")
TRAIN_PREPROC_DIR = Path("preprocessed")
MODEL_PATH = Path("preprocessed/mlp_whale_classifier.pt")
WORK_DIR = Path("testing_artifacts")
BATCH_SIZE = 256
CONFUSION_MATRIX_IMAGE_PATH = Path("confusion_matrix_testing.png")


def print_confusion_matrix(cm: np.ndarray, label_names: np.ndarray) -> None:
    print("Confusion Matrix (counts) | rows=true, cols=pred")
    header = "true\\pred".ljust(12) + " ".join(str(name).rjust(9) for name in label_names)
    print(header)
    for i, name in enumerate(label_names):
        row = " ".join(str(int(v)).rjust(9) for v in cm[i])
        print(str(name).ljust(12) + row)

    print("\nConfusion Matrix (row-normalized)")
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    header_norm = "true\\pred".ljust(12) + " ".join(str(name).rjust(9) for name in label_names)
    print(header_norm)
    for i, name in enumerate(label_names):
        row = " ".join(f"{v:.2f}".rjust(9) for v in cm_norm[i])
        print(str(name).ljust(12) + row)


def save_confusion_matrix_image(cm: np.ndarray, label_names: np.ndarray, output_path: Path) -> None:
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title("Validation Confusion Matrix (Normalized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    indices = np.arange(len(label_names))
    ax.set_xticks(indices)
    ax.set_yticks(indices)
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]}\n{cm_norm[i, j]:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix image to: {output_path}")


def main() -> None:
    train_stats_path = TRAIN_PREPROC_DIR / "normalization_stats.npz"
    train_label_names_path = TRAIN_PREPROC_DIR / "label_names.npy"

    for path in [train_stats_path, train_label_names_path, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    testing_validation_dir = WORK_DIR / "validation_testing"
    testing_preproc_dir = testing_validation_dir / "testing_preprocessed"

    preprocess_result = preprocess_split(
        dataset_root=DATASET_ROOT,
        split="validation",
        events_dir=testing_validation_dir / "testing_waves",
        array_output_path=testing_validation_dir / "testing_spectrograms_arrays.npy",
        index_output_path=testing_validation_dir / "testing_spectrograms_arrays_index.csv",
        preproc_dir=testing_preproc_dir,
        train_stats_path=train_stats_path,
        label_names_path=train_label_names_path,
    )

    X_val = np.load(preprocess_result.normalized_path, mmap_mode="r")
    y_val = np.load(preprocess_result.labels_path)
    label_names = np.load(preprocess_result.label_names_path)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    checkpoint_input_shape = tuple(int(v) for v in checkpoint["input_shape"])
    if tuple(X_val.shape[1:]) != checkpoint_input_shape:
        raise ValueError(
            f"Input shape mismatch. Validation={tuple(X_val.shape[1:])}, model expects={checkpoint_input_shape}."
        )

    model_label_names = [str(v) for v in checkpoint["label_names"]]
    if model_label_names != [str(v) for v in label_names]:
        raise ValueError("Label order mismatch between model checkpoint and validation preprocessing.")

    num_classes = len(label_names)
    input_dim = int(np.prod(checkpoint_input_shape))
    config = checkpoint.get("config", {})
    hidden_1 = int(config.get("hidden_1", 256))
    hidden_2 = int(config.get("hidden_2", 128))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(input_dim, num_classes, hidden_1=hidden_1, hidden_2=hidden_2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = SpectrogramDataset(X_val, y_val)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            preds = model(xb).argmax(dim=1).cpu().numpy()
            true = yb.cpu().numpy()
            np.add.at(cm, (true, preds), 1)

    accuracy = float(np.trace(cm) / max(cm.sum(), 1))
    print(f"Validation samples: {len(y_val)}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print_confusion_matrix(cm, label_names)
    save_confusion_matrix_image(cm, label_names, CONFUSION_MATRIX_IMAGE_PATH)


if __name__ == "__main__":
    main()
