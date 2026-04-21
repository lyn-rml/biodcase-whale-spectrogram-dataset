#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.lib.format import open_memmap
from scipy.io import wavfile


NFFT = 512
WINSIZE = 256
OVERLAP_PERCENT = 98
SCALE = "default"

# Hardcoded execution constants
DATASET_ROOT = Path("biodcase_development_set")
SPLIT = "train"
WORK_DIR = Path("artifacts")
PREPROC_DIR = Path("preprocessed")

FREQ_PERCENTILE = 90
TIME_PERCENTILE = 90
MAX_FREQ_BINS = 128
MAX_TIME_BINS = 256
ROUND_FREQ_TO = 8
ROUND_TIME_TO = 16

TRAIN_STATS_PATH: Path | None = None
LABEL_NAMES_PATH: Path | None = None


@dataclass
class PreprocessResult:
    split: str
    events_dir: Path
    array_output_path: Path
    index_output_path: Path
    resized_path: Path
    normalized_path: Path
    labels_path: Path
    label_names_path: Path
    stats_path: Path
    frequencies_path: Path
    frequencies_stats_path: Path
    num_samples: int
    target_freq_bins: int
    target_time_bins: int


def _parse_iso_datetime(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).replace(tzinfo=None)


def compute_event_window_ms(row: dict[str, str]) -> tuple[float, float]:
    recording_start_token = Path(row["filename"]).stem.split("_")[0]
    recording_start = datetime.strptime(recording_start_token, "%Y-%m-%dT%H-%M-%S")

    start_dt = _parse_iso_datetime(row["start_datetime"])
    end_dt = _parse_iso_datetime(row["end_datetime"])

    start_ms = (start_dt - recording_start).total_seconds() * 1000.0
    end_ms = (end_dt - recording_start).total_seconds() * 1000.0
    return start_ms, end_ms


def iter_annotation_rows(annotations_dir: Path) -> Iterable[dict[str, str]]:
    for csv_path in sorted(annotations_dir.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row


def reset_wav_dir(events_dir: Path) -> None:
    events_dir.mkdir(parents=True, exist_ok=True)
    for wav_path in events_dir.glob("*.wav"):
        wav_path.unlink()


def extract_events_to_wavs(dataset_root: Path, split: str, events_dir: Path) -> int:
    annotations_dir = dataset_root / split / "annotations"
    audio_root = dataset_root / split / "audio"

    if not annotations_dir.exists():
        raise FileNotFoundError(f"Missing annotations directory: {annotations_dir}")
    if not audio_root.exists():
        raise FileNotFoundError(f"Missing audio directory: {audio_root}")

    reset_wav_dir(events_dir)

    event_index = 0
    extracted = 0

    last_audio_path: Path | None = None
    last_sr: int | None = None
    last_data: np.ndarray | None = None

    for row in iter_annotation_rows(annotations_dir):
        event_index += 1

        wav_name = row["filename"].strip()
        dataset_name = row["dataset"].strip()
        src_path = audio_root / dataset_name / wav_name
        if not src_path.exists():
            continue

        if src_path != last_audio_path:
            sr, data = wavfile.read(src_path)
            last_audio_path = src_path
            last_sr = int(sr)
            last_data = data

        if last_sr is None or last_data is None:
            continue

        start_ms, end_ms = compute_event_window_ms(row)
        start_idx = max(0, int(round(start_ms * last_sr / 1000.0)))
        end_idx = min(last_data.shape[0], int(round(end_ms * last_sr / 1000.0)))

        if end_idx <= start_idx:
            continue

        segment = last_data[start_idx:end_idx]

        annotation = row["annotation"].strip()
        low_freq = row["low_frequency"].strip()
        high_freq = row["high_frequency"].strip()
        event_name = f"{annotation}_{low_freq}_{high_freq}_{event_index}.wav"

        wavfile.write(events_dir / event_name, last_sr, segment)
        extracted += 1

    return extracted


def convert_wav_to_float(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.uint8:
        data = (data - 128) / 128.0
    elif data.dtype == np.int16:
        data = data / 32768.0
    elif data.dtype == np.int32:
        data = data / 2147483648.0
    return data.astype(np.float32, copy=False)


def parse_event_filename(stem: str) -> tuple[str, float, float]:
    parts = stem.rsplit("_", 3)
    if len(parts) != 4:
        raise ValueError(f"Unexpected event filename format: {stem}")

    label, low_str, high_str, _ = parts
    return label, float(low_str), float(high_str)


def load_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    data = convert_wav_to_float(data)

    if data.ndim > 1:
        data = data.mean(axis=1)

    return int(sample_rate), data


def compute_spectrogram_array(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from matplotlib import mlab

    noverlap = int(round(WINSIZE * (OVERLAP_PERCENT / 100.0)))
    noverlap = min(noverlap, WINSIZE - 1)

    power, freqs, times = mlab.specgram(
        x=signal,
        NFFT=WINSIZE,
        Fs=sample_rate,
        noverlap=noverlap,
        pad_to=NFFT,
    )

    if SCALE in {"default", "dB"}:
        power = 10.0 * np.log10(np.maximum(power, np.finfo(np.float32).tiny))

    return (
        power.astype(np.float32, copy=False),
        freqs.astype(np.float32, copy=False),
        times.astype(np.float32, copy=False),
    )


def append_event_arrays(stream_file, csv_writer: csv.writer, record_index: int, wav_path: Path) -> None:
    label, low_freq, high_freq = parse_event_filename(wav_path.stem)
    sample_rate, signal = load_wav_mono(wav_path)
    spectrogram, freqs, times = compute_spectrogram_array(signal, sample_rate)

    low = min(low_freq, high_freq)
    high = max(low_freq, high_freq)
    band_mask = (freqs >= low) & (freqs <= high)

    if not np.any(band_mask):
        nearest_idx = int(np.argmin(np.abs(freqs - ((low + high) / 2.0))))
        band_mask = np.zeros_like(freqs, dtype=bool)
        band_mask[nearest_idx] = True

    spectrogram = spectrogram[band_mask, :]
    freqs = freqs[band_mask]

    spec_offset = stream_file.tell()
    np.save(stream_file, spectrogram, allow_pickle=False)
    freq_offset = stream_file.tell()
    np.save(stream_file, freqs, allow_pickle=False)
    time_offset = stream_file.tell()
    np.save(stream_file, times, allow_pickle=False)

    csv_writer.writerow(
        [
            record_index,
            wav_path.name,
            label,
            low,
            high,
            sample_rate,
            spectrogram.shape[0],
            spectrogram.shape[1],
            spec_offset,
            freq_offset,
            time_offset,
        ]
    )


def build_spectrogram_stream(events_dir: Path, array_output_path: Path, index_output_path: Path) -> int:
    wav_files = sorted(events_dir.glob("*.wav"))

    array_output_path.parent.mkdir(parents=True, exist_ok=True)
    index_output_path.parent.mkdir(parents=True, exist_ok=True)

    with array_output_path.open("wb") as stream_file, index_output_path.open("w", newline="", encoding="utf-8") as index_file:
        writer = csv.writer(index_file)
        writer.writerow(
            [
                "record_index",
                "wav_filename",
                "label",
                "low_frequency",
                "high_frequency",
                "sample_rate",
                "n_frequency_bins",
                "n_time_bins",
                "spectrogram_offset",
                "frequencies_offset",
                "times_offset",
            ]
        )

        for record_index, wav_path in enumerate(wav_files):
            append_event_arrays(stream_file, writer, record_index, wav_path)

    return len(wav_files)


def load_index_rows(index_path: Path) -> list[dict[str, str]]:
    with index_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Index file is empty: {index_path}")

    return rows


def _round_up(value: int, base: int) -> int:
    return int(np.ceil(value / base) * base)


def select_target_shape(
    rows: list[dict[str, str]],
    freq_percentile: int,
    time_percentile: int,
    max_freq_bins: int,
    max_time_bins: int,
    round_freq_to: int,
    round_time_to: int,
) -> tuple[int, int]:
    freq_bins = np.array([int(r["n_frequency_bins"]) for r in rows], dtype=np.int32)
    time_bins = np.array([int(r["n_time_bins"]) for r in rows], dtype=np.int32)

    p_freq = int(np.percentile(freq_bins, freq_percentile))
    p_time = int(np.percentile(time_bins, time_percentile))

    target_freq_bins = min(max_freq_bins, _round_up(max(1, p_freq), round_freq_to))
    target_time_bins = min(max_time_bins, _round_up(max(1, p_time), round_time_to))
    return target_freq_bins, target_time_bins


def resize_2d_bilinear(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = x.shape
    if in_h == out_h and in_w == out_w:
        return x.astype(np.float32, copy=True)

    x = x.astype(np.float32, copy=False)

    old_x = np.linspace(0.0, 1.0, in_w, dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, out_w, dtype=np.float32)
    tmp = np.empty((in_h, out_w), dtype=np.float32)
    for i in range(in_h):
        tmp[i, :] = np.interp(new_x, old_x, x[i, :])

    old_y = np.linspace(0.0, 1.0, in_h, dtype=np.float32)
    new_y = np.linspace(0.0, 1.0, out_h, dtype=np.float32)
    out = np.empty((out_h, out_w), dtype=np.float32)
    for j in range(out_w):
        out[:, j] = np.interp(new_y, old_y, tmp[:, j])

    return out


def load_spectrogram(stream_file, spectrogram_offset: int) -> np.ndarray:
    stream_file.seek(spectrogram_offset)
    return np.load(stream_file, allow_pickle=False).astype(np.float32, copy=False)


def build_resized_arrays(
    array_path: Path,
    rows: list[dict[str, str]],
    resized_path: Path,
    labels_path: Path,
    label_names_path: Path,
    target_shape: tuple[int, int],
    fixed_label_names: Sequence[str] | None,
) -> tuple[np.ndarray, int]:
    target_freq_bins, target_time_bins = target_shape
    num_samples = len(rows)

    resized_path.parent.mkdir(parents=True, exist_ok=True)
    X_resized = open_memmap(
        resized_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, target_freq_bins, target_time_bins),
    )

    if fixed_label_names is None:
        label_names = sorted({r["label"] for r in rows})
    else:
        label_names = [str(v) for v in fixed_label_names]
        unknown = sorted({r["label"] for r in rows} - set(label_names))
        if unknown:
            raise ValueError(f"Unknown labels not present in provided label names: {unknown}")

    label_to_id = {name: idx for idx, name in enumerate(label_names)}
    y = np.empty(num_samples, dtype=np.int32)

    with array_path.open("rb") as stream_file:
        for i, row in enumerate(rows):
            spec = load_spectrogram(stream_file, int(row["spectrogram_offset"]))
            X_resized[i] = resize_2d_bilinear(spec, target_freq_bins, target_time_bins)
            y[i] = label_to_id[row["label"]]

            if (i + 1) % 2000 == 0:
                print(f"Resized {i + 1}/{num_samples}")

    X_resized.flush()
    np.save(labels_path, y)
    np.save(label_names_path, np.array(label_names, dtype="U32"))
    return np.array(label_names), num_samples


def compute_mean_std(x_path: Path) -> tuple[float, float]:
    X_resized = np.load(x_path, mmap_mode="r")
    num_samples = X_resized.shape[0]

    sum_x = np.float64(0.0)
    sum_x2 = np.float64(0.0)
    count = np.int64(0)

    for i in range(num_samples):
        arr = X_resized[i]
        sum_x += arr.sum(dtype=np.float64)
        sum_x2 += np.square(arr, dtype=np.float64).sum(dtype=np.float64)
        count += arr.size

    mean = float(sum_x / count)
    var = float((sum_x2 / count) - (mean ** 2))
    std = float(np.sqrt(max(var, 1e-12)))
    return mean, std


def normalize_arrays(
    resized_path: Path,
    normalized_path: Path,
    stats_path: Path,
    target_shape: tuple[int, int],
    provided_mean_std: tuple[float, float] | None,
) -> tuple[float, float]:
    X_resized = np.load(resized_path, mmap_mode="r")
    num_samples = X_resized.shape[0]

    if provided_mean_std is None:
        mean, std = compute_mean_std(resized_path)
    else:
        mean, std = provided_mean_std

    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    X_norm = open_memmap(
        normalized_path,
        mode="w+",
        dtype=np.float32,
        shape=X_resized.shape,
    )

    for i in range(num_samples):
        X_norm[i] = (X_resized[i] - mean) / std
        if (i + 1) % 2000 == 0:
            print(f"Normalized {i + 1}/{num_samples}")

    X_norm.flush()
    np.savez(
        stats_path,
        mean=np.float32(mean),
        std=np.float32(std),
        target_freq_bins=np.int32(target_shape[0]),
        target_time_bins=np.int32(target_shape[1]),
        num_samples=np.int32(num_samples),
    )
    return mean, std


def extract_and_save_frequency_features(
    rows: list[dict[str, str]],
    freqs_path: Path,
    freqs_stats_path: Path,
    provided_freqs_stats: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Extract low and high frequency from index rows and save as normalized features.
    
    Returns:
        - Frequency features array of shape (num_samples, 2)
        - Frequency statistics (low_min, low_max, high_min, high_max) for normalization
    """
    num_samples = len(rows)
    frequencies = np.empty((num_samples, 2), dtype=np.float32)
    
    for i, row in enumerate(rows):
        low_freq = float(row["low_frequency"])
        high_freq = float(row["high_frequency"])
        frequencies[i] = [low_freq, high_freq]
    
    # Compute or use provided frequency statistics
    if provided_freqs_stats is None:
        low_min = float(frequencies[:, 0].min())
        low_max = float(frequencies[:, 0].max())
        high_min = float(frequencies[:, 1].min())
        high_max = float(frequencies[:, 1].max())
    else:
        low_min, low_max, high_min, high_max = provided_freqs_stats
    
    # Normalize frequencies to [0, 1]
    frequencies[:, 0] = (frequencies[:, 0] - low_min) / (low_max - low_min + 1e-9)
    frequencies[:, 1] = (frequencies[:, 1] - high_min) / (high_max - high_min + 1e-9)
    
    freqs_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(freqs_path, frequencies)
    
    freqs_stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        freqs_stats_path,
        low_min=np.float32(low_min),
        low_max=np.float32(low_max),
        high_min=np.float32(high_min),
        high_max=np.float32(high_max),
    )
    
    return frequencies, (low_min, low_max, high_min, high_max)


def preprocess_split(
    dataset_root: Path,
    split: str,
    events_dir: Path,
    array_output_path: Path,
    index_output_path: Path,
    preproc_dir: Path,
    freq_percentile: int = 90,
    time_percentile: int = 90,
    max_freq_bins: int = 128,
    max_time_bins: int = 256,
    round_freq_to: int = 8,
    round_time_to: int = 16,
    train_stats_path: Path | None = None,
    label_names_path: Path | None = None,
    train_freqs_stats_path: Path | None = None,
) -> PreprocessResult:
    preproc_dir.mkdir(parents=True, exist_ok=True)

    resized_path = preproc_dir / "X_resized.npy"
    normalized_path = preproc_dir / "X_normalized.npy"
    labels_path = preproc_dir / "y_labels.npy"
    out_label_names_path = preproc_dir / "label_names.npy"
    stats_path = preproc_dir / "normalization_stats.npz"
    frequencies_path = preproc_dir / "frequencies.npy"
    frequencies_stats_path = preproc_dir / "frequencies_stats.npz"

    extracted = extract_events_to_wavs(dataset_root, split, events_dir)
    print(f"Extracted event WAV files: {extracted}")

    stream_count = build_spectrogram_stream(events_dir, array_output_path, index_output_path)
    print(f"Spectrogram records written: {stream_count}")

    rows = load_index_rows(index_output_path)

    fixed_target_shape: tuple[int, int] | None = None
    provided_mean_std: tuple[float, float] | None = None
    fixed_label_names: Sequence[str] | None = None
    provided_freqs_stats: tuple[float, float, float, float] | None = None

    if train_stats_path is not None:
        stats = np.load(train_stats_path)
        fixed_target_shape = (int(stats["target_freq_bins"]), int(stats["target_time_bins"]))
        provided_mean_std = (float(stats["mean"]), float(stats["std"]))

    if label_names_path is not None:
        fixed_label_names = [str(v) for v in np.load(label_names_path)]

    if train_freqs_stats_path is not None:
        freqs_stats = np.load(train_freqs_stats_path)
        provided_freqs_stats = (
            float(freqs_stats["low_min"]),
            float(freqs_stats["low_max"]),
            float(freqs_stats["high_min"]),
            float(freqs_stats["high_max"]),
        )

    target_shape = fixed_target_shape or select_target_shape(
        rows=rows,
        freq_percentile=freq_percentile,
        time_percentile=time_percentile,
        max_freq_bins=max_freq_bins,
        max_time_bins=max_time_bins,
        round_freq_to=round_freq_to,
        round_time_to=round_time_to,
    )

    print(f"Target spectrogram shape: {target_shape}")

    label_names, num_samples = build_resized_arrays(
        array_path=array_output_path,
        rows=rows,
        resized_path=resized_path,
        labels_path=labels_path,
        label_names_path=out_label_names_path,
        target_shape=target_shape,
        fixed_label_names=fixed_label_names,
    )

    mean, std = normalize_arrays(
        resized_path=resized_path,
        normalized_path=normalized_path,
        stats_path=stats_path,
        target_shape=target_shape,
        provided_mean_std=provided_mean_std,
    )

    frequencies, freqs_stats = extract_and_save_frequency_features(
        rows=rows,
        freqs_path=frequencies_path,
        freqs_stats_path=frequencies_stats_path,
        provided_freqs_stats=provided_freqs_stats,
    )

    print(f"Saved labels: {labels_path}")
    print(f"Saved class names: {out_label_names_path}")
    print(f"Saved normalized arrays: {normalized_path}")
    print(f"Saved frequency features: {frequencies_path}")
    print(f"Normalization mean={mean:.6f}, std={std:.6f}")
    print(f"Classes: {list(label_names)}")

    return PreprocessResult(
        split=split,
        events_dir=events_dir,
        array_output_path=array_output_path,
        index_output_path=index_output_path,
        resized_path=resized_path,
        normalized_path=normalized_path,
        labels_path=labels_path,
        label_names_path=out_label_names_path,
        stats_path=stats_path,
        frequencies_path=frequencies_path,
        frequencies_stats_path=frequencies_stats_path,
        num_samples=num_samples,
        target_freq_bins=target_shape[0],
        target_time_bins=target_shape[1],
    )


def main() -> None:
    split_dir = WORK_DIR / SPLIT
    events_dir = split_dir / "waves"
    array_output_path = split_dir / "spectrograms_arrays.npy"
    index_output_path = split_dir / "spectrograms_arrays_index.csv"

    preproc_dir = PREPROC_DIR

    preprocess_split(
        dataset_root=DATASET_ROOT,
        split=SPLIT,
        events_dir=events_dir,
        array_output_path=array_output_path,
        index_output_path=index_output_path,
        preproc_dir=preproc_dir,
        freq_percentile=FREQ_PERCENTILE,
        time_percentile=TIME_PERCENTILE,
        max_freq_bins=MAX_FREQ_BINS,
        max_time_bins=MAX_TIME_BINS,
        round_freq_to=ROUND_FREQ_TO,
        round_time_to=ROUND_TIME_TO,
        train_stats_path=TRAIN_STATS_PATH,
        label_names_path=LABEL_NAMES_PATH,
    )


if __name__ == "__main__":
    main()
