# Biodcase Whale Spectrogram Dataset

A whale audio classification project using spectrograms and neural networks.

## Project Structure

### Core Files

- **`preprocessing_wavfiles.py`** – Converts WAV audio files from the training dataset into spectrograms with normalization and resizing
- **`training.py`** – Defines the MLP neural network and handles model training on preprocessed spectrograms
- **`testing.py`** – Loads the trained model and evaluates it on validation data, generates confusion matrices
- **`main.ipynb`** – Development notebook with utility functions for audio processing (testing area)

### Data Directories

- **`biodcase_development_set/`**
  - `train/` – Training data with annotations (CSV) and audio files (WAV)
  - `validation/` – Validation data with annotations and audio files
- **`preprocessed/`** – Generated spectrogram arrays and model weights
- **`spectrograms/`** – Spectrogram image outputs
- **`ML_images/`** – ML visualization outputs
- **`spectrogram_samples/`** – Sample spectrograms for reference

## Workflow

1. Run `preprocessing_wavfiles.py` to process training audio files into spectrograms
2. Run `training.py` to train the classifier on preprocessed data
3. Run `testing.py` to validate the model on unseen validation data

### Presentation
 - The main presentation `Classification-Acoustique-des-Vocalisations-de-Baleines_EDIT.pptx` contains an explanation for the methods and pipeline that was implemented in this project