#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:
    raise ImportError("PyTorch is not installed in this environment.") from exc

from preprocessing_wavfiles import preprocess_split
from training import MLPClassifier, SpectrogramDataset


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess validation split and evaluate saved model with confusion matrix.")
    parser.add_argument("--dataset-root", type=Path, default=Path("biodcase_development_set"))
    parser.add_argument("--train-preproc-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--model-path", type=Path, default=Path("preprocessed/mlp_whale_classifier.pt"))
    parser.add_argument("--work-dir", type=Path, default=Path("testing_artifacts"))
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_stats_path = args.train_preproc_dir / "normalization_stats.npz"
    train_label_names_path = args.train_preproc_dir / "label_names.npy"

    for path in [train_stats_path, train_label_names_path, args.model_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    testing_validation_dir = args.work_dir / "validation_testing"
    testing_preproc_dir = testing_validation_dir / "testing_preprocessed"

    preprocess_result = preprocess_split(
        dataset_root=args.dataset_root,
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

    checkpoint = torch.load(args.model_path, map_location="cpu")
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
        batch_size=args.batch_size,
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


if __name__ == "__main__":
    main()
