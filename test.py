import csv
from pathlib import Path

import numpy as np
from matplotlib import mlab
from scipy.io import wavfile


WAV_DIR = Path("./waves")
ARRAY_OUTPUT_PATH = Path("./spectrograms_arrays.npy")
INDEX_OUTPUT_PATH = Path("./spectrograms_arrays_index.csv")

NFFT = 512
WINSIZE = 256
OVERLAP_PERCENT = 98
SCALE = "default"


def convert_wav_to_float(data: np.ndarray) -> np.ndarray:
    """Match the notebook conversion so plotting behaves consistently across WAV dtypes."""
    if data.dtype == np.uint8:
        data = (data - 128) / 128.0
    elif data.dtype == np.int16:
        data = data / 32768.0
    elif data.dtype == np.int32:
        data = data / 2147483648.0
    return data.astype(np.float32, copy=False)


def parse_filename(stem: str) -> tuple[str, float, float]:
    """Expected pattern: <label>_<low_freq>_<high_freq>_<index>.wav"""
    parts = stem.split("_")
    if len(parts) != 4:
        raise ValueError(f"Unexpected filename format: {stem}")

    label, low_str, high_str, _ = parts
    return label, float(low_str), float(high_str)


def load_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    data = convert_wav_to_float(data)

    if data.ndim > 1:
        data = data.mean(axis=1)

    return sample_rate, data


def compute_spectrogram_array(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep spectrogram math aligned with matplotlib.specgram settings:
    nfft=512 | winsize=256 | overlap=98 | scale=default.
    """
    noverlap = int(round(WINSIZE * (OVERLAP_PERCENT / 100.0)))
    noverlap = min(noverlap, WINSIZE - 1)

    power, freqs, times = mlab.specgram(
        x=signal,
        NFFT=WINSIZE,
        Fs=sample_rate,
        noverlap=noverlap,
        pad_to=NFFT,
    )

    if SCALE == "default" or SCALE == "dB":
        # "default" uses dB for PSD spectrograms.
        power = 10.0 * np.log10(np.maximum(power, np.finfo(np.float32).tiny))

    return (
        power.astype(np.float32, copy=False),
        freqs.astype(np.float32, copy=False),
        times.astype(np.float32, copy=False),
    )


def append_event_arrays(
    stream_file,
    csv_writer: csv.writer,
    record_index: int,
    wav_path: Path,
) -> None:
    label, low_freq, high_freq = parse_filename(wav_path.stem)
    sample_rate, signal = load_wav_mono(wav_path)
    spectrogram, freqs, times = compute_spectrogram_array(signal, sample_rate)

    low = min(low_freq, high_freq)
    high = max(low_freq, high_freq)
    band_mask = (freqs >= low) & (freqs <= high)
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


def main() -> None:
    wav_files = sorted(WAV_DIR.glob("*.wav"))

    with ARRAY_OUTPUT_PATH.open("wb") as stream_file, INDEX_OUTPUT_PATH.open("w", newline="") as index_file:
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


if __name__ == "__main__":
    main()

# Summary:
# - Iterates over every WAV in ./waves.
# - Parses filename as <label>_<low_freq>_<high_freq>_<index>.wav.
# - Builds one spectrogram per WAV using nfft=512, winsize=256, overlap=98, scale=default.
# - Crops each spectrogram to its filename low/high frequency band.
# - Appends spectrogram/frequency/time arrays into one NumPy binary stream: ./spectrograms_arrays.npy
# - Writes an index CSV (offsets + metadata) to ./spectrograms_arrays_index.csv for retrieval.
