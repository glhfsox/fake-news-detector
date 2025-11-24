# Fake News Detector (Baseline: TF-IDF + Logistic Regression)

This repository contains a simple baseline model for fake news detection.
The goal is to classify news articles as **REAL** or **FAKE** based on their text.

The current version includes:
- data preparation script (merging True/Fake CSVs, creating train/val/test),
- baseline model: TF-IDF + Logistic Regression,
- a simple CLI tool for interactive predictions.

---

## Project structure

fake-news-detector/
  data/
    raw/
      True.csv        # original real news dataset
      Fake.csv        # original fake news dataset
    processed/
      train.csv       # prepared train split
      val.csv         # prepared validation split
      test.csv        # prepared test split
  src/
    data/
      prepare.py      # data loading, merging, splitting into train/val/test
    models/
      baseline.py     # training and evaluation of TF-IDF + LogisticRegression
      predict_baseline.py  # CLI for interactive predictions
  .venv/              # (optional) virtual environment, not tracked by git
  README.md
  requirements.txt


Requirements

Python 3.10+ (recommended)

pip (or other package manager)

Python dependencies (can be installed via requirements.txt):

pandas

numpy

scikit-learn

joblib

Data preparation

Place the original datasets in: 

data/raw/True.csv
data/raw/Fake.csv


Notes and limitations

This is a baseline model based on TF-IDF and Logistic Regression.

The model classifies texts by style and vocabulary, not by checking facts.

It is trained on a specific fake/real news dataset (True.csv / Fake.csv), so behavior on other domains may differ.

Future versions will include transformer-based models (e.g. DistilBERT) and a web interface.


## Version 2: DistilBERT classifier

- Training: `python -m src.models.transformers` (saves model and tokenizer to `src/models/transformer-distilbert`).
- Interactive predictions: `python -m src.models.predict_transformer`.

Notes:
- Labels: 0 → REAL, 1 → FAKE.
- Uses max_length=256 and pads/truncates to that length.
- Runs on GPU if CUDA, otherwise CPU.
