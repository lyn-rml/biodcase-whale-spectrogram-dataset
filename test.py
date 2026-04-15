from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


WAV_DIR = Path("./waves")
OUTPUT_DIR = Path("./spectrograms")

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
    """Expected pattern: <label>_<low_freq>_<high_freq>.wav"""
    parts = stem.split("_")
    if len(parts) != 4:
        raise ValueError(f"Unexpected filename format: {stem}")

    label, low_str, high_str, index= parts
    return label, float(low_str), float(high_str)


def load_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    data = convert_wav_to_float(data)

    # Convert multi-channel WAV to mono for a single spectrogram image.
    if data.ndim > 1:
        data = data.mean(axis=1)

    return sample_rate, data


def save_spectrogram(wav_path: Path) -> None:
    _, low_freq, high_freq = parse_filename(wav_path.stem)
    sample_rate, signal = load_wav_mono(wav_path)

    noverlap = int(round(WINSIZE * (OVERLAP_PERCENT / 100.0)))
    noverlap = min(noverlap, WINSIZE - 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.specgram(
        signal,
        NFFT=WINSIZE,
        Fs=sample_rate,
        noverlap=noverlap,
        pad_to=NFFT,
        scale=SCALE,
        cmap="cividis",
    )

    low = min(low_freq, high_freq)
    high = max(low_freq, high_freq)
    ax.set_ylim(low, high)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{wav_path.stem}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    wav_files = sorted(WAV_DIR.glob("*.wav"))
    for wav_path in wav_files:
        save_spectrogram(wav_path)


if __name__ == "__main__":
    main()

# Summary:
# - Iterates over every WAV in ./waves.
# - Parses filename as <label>_<low_freq>_<high_freq>.wav.
# - Builds one spectrogram per WAV using nfft=512, winsize=256, overlap=98, scale=default.
# - Uses the parsed low/high frequencies as the y-axis limits.
# - Removes titles/labels/axes so each output image is a clean spectrogram only.
# - Saves each image to ./spectrograms with the same base filename and .png extension.
