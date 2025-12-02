import os
import re
from collections import Counter
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "src/models/transformer-distilbert"
TRAIN_PATH = "data/processed/train.csv"
LABEL_TO_NAME = {0: "REAL", 1: "FAKE"}


@st.cache_resource
def load_model():
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource
def load_word_stats():
    df = pd.read_csv(TRAIN_PATH)
    # ожидаем колонки full_text, label
    if "full_text" not in df.columns:
        # если структура другая, соберём full_text из title+text
        if "title" in df.columns and "text" in df.columns:
            title = df["title"].fillna("").astype(str)
            text = df["text"].fillna("").astype(str)
            df["full_text"] = title.str.cat(text, sep=" ")
        else:
            raise ValueError("No full_text or title/text columns in train.csv")


    df["full_text"] = df["full_text"].astype(str)
    df["label"] = df["label"].astype(int)

    token_re = re.compile(r"\w+", re.UNICODE)
    freq_real = Counter()
    freq_fake = Counter()

    for _, row in df.iterrows():
        tokens = token_re.findall(row["full_text"].lower())
        if row["label"] == 0:
            freq_real.update(tokens)
        else:
            freq_fake.update(tokens)

    return freq_real, freq_fake


def predict_text(
    tokenizer, model, device, text: str
) -> Tuple[int, str, float, float]:
    if not text.strip():
        raise ValueError("Empty text")

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


def color_text(text: str, freq_real: Counter, freq_fake: Counter,
               alpha: float = 1.0, threshold: float = 0.5) -> str:
    parts = re.findall(r"\w+|\s+|[^\w\s]", text, flags=re.UNICODE)
    html_parts = []

    for tok in parts:
        if not tok.strip() or not tok.isalpha():
            html_parts.append(tok)
            continue

        w = tok.lower()
        fr = freq_real[w]
        ff = freq_fake[w]
        score = np.log((ff + alpha) / (fr + alpha))

        if score > threshold:
            color = "#e74c3c"  
        elif score < -threshold:
            color = "#2ecc71"  
        else:
            html_parts.append(tok)
            continue

        html_parts.append(
            f'<span style="color:{color}; font-weight:600">{tok}</span>'
        )

    return "".join(html_parts)


def main():
    st.set_page_config(
        page_title="Fake News Detector",
        layout="wide",
    )

    tokenizer, model, device = load_model()
    freq_real, freq_fake = load_word_stats()

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Type, paste, or upload your text")
        text = st.text_area(
            "News text",
            height=400,
            label_visibility="collapsed",
        )
        run_button = st.button("Scan for fake news")

    with right:
        st.subheader("Detection result")

        if run_button and text.strip():
            label, label_name, real_p, fake_p = predict_text(
                tokenizer, model, device, text
            )

            fake_pct = fake_p * 100
            real_pct = real_p * 100

            st.markdown(
                f"### {fake_pct:.1f}% probability this text is **FAKE**"
            )

            # Круговой индикатор вероятности FAKE
            color_fake = "#e74c3c"
            color_real = "#2ecc71"

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=fake_pct,
                    number={"suffix": "%"},
                    title={"text": "FAKE probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color_fake},
                        "steps": [
                            {"range": [0, 50], "color": color_real},
                            {"range": [50, 100], "color": color_fake},
                        ],
                    },
                )
            )
            fig.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                f"- REAL: **{real_pct:.1f}%**  \n"
                f"- FAKE: **{fake_pct:.1f}%**"
            )
        else:
            st.write("Enter text on the left and press **Scan**.")

    st.markdown("---")
    st.markdown("### Token highlights")

    if run_button and text.strip():
        colored_html = color_text(text, freq_real, freq_fake)
        st.markdown(colored_html, unsafe_allow_html=True)
    else:
        st.write("Words more typical for FAKE will be red, for REAL — green.")


if __name__ == "__main__":
    main()
