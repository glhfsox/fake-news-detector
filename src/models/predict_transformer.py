import os
from typing import Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "src/models/transformer-distilbert"
LABEL_TO_NAME = {0: "REAL", 1: "FAKE"}


def load_model(model_dir: str = MODEL_DIR):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading transformer model from {model_dir} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_text(
    tokenizer, model, device, text: str
) -> Tuple[int, str, float, float]:
    if not text or not text.strip():
        raise ValueError("Empty text. Please enter news text.")

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    real_prob = float(probs[0])
    fake_prob = float(probs[1])
    label = int(np.argmax(probs))
    label_name = LABEL_TO_NAME.get(label, str(label))

    return label, label_name, real_prob, fake_prob


def main():
    tokenizer, model, device = load_model()

    print("\nFake News Detector — DistilBERT (transformer)")
    print("Enter news text. Type 'q' or press Enter on empty line to quit.\n")

    while True:
        text = input("News: ").strip()
        if text.lower() in {"q", "quit", "exit"} or text == "":
            print("Exiting...")
            break

        try:
            label, label_name, real_p, fake_p = predict_text(
                tokenizer, model, device, text
            )
        except ValueError as e:
            print(f"Error: {e}")
            continue

        print(f"\nPredicted class: {label_name} (label = {label})")
        print(f"Probability REAL: {real_p * 100:.3f} %")
        print(f"Probability FAKE: {fake_p * 100:.3f} %\n")

        again = input("Try another? (y/n): ").strip().lower()
        if again not in {"y", "yes"}:
            print("Exiting...")
            break


if __name__ == "__main__":
    main()
