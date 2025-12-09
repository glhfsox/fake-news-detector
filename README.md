# Fake News Detector

Classifies news as **REAL** or **FAKE**. Two models:
- Baseline: TF‑IDF + Logistic Regression
- V2: DistilBERT transformer

## Data
Sources used now (adjust in `src/data/prepare.py`):
- Real: `data/raw/True.csv`, `data/raw/DataSet_Misinfo_TRUE.csv`
- Fake: `data/raw/Fake.csv`, `data/raw/DataSet_Misinfo_FAKE.csv`, `data/raw/EXTRA_RussianPropagandaSubset.csv`
- Mixed (has labels): `data/raw/welfake_dataset.csv`

## Preparation
1) Drop raw CSVs into `data/raw/`.
2) Run: `python -m src.data.prepare`
   - Outputs stratified splits to `data/processed/train.csv`, `val.csv`, `test.csv`
   - Dedups texts (case/whitespace normalized), drops conflicting labels, handles files without `title`.

## Training
- Baseline: `python -m src.models.baseline` → `src/models/baseline_tfidf_logreg.joblib`
- DistilBERT: `python -m src.models.transformers`
  - Saves model/tokenizer + `metrics.json` to `src/models/transformer-distilbert`

## Prediction (CLI)
- Baseline: `python -m src.models.predict_baseline`
- DistilBERT: `python -m src.models.predict_transformer`
  - Labels: 0 → REAL, 1 → FAKE
  - Uses CUDA if available, otherwise CPU

## Web UI.
- Run: `streamlit run src/ui/fake_news_app.py`
  - Left: text area for pasting news.
  - Right: circular gauge showing FAKE probability and REAL/FAKE percentages.
  - Below: words more typical for FAKE highlighted red, for REAL highlighted green.

## Quick test texts
REAL:
- “The European Space Agency confirmed its JUICE probe performed a gravity assist around Earth and will reach Jupiter’s moons in 2031 as planned.”
- “NASA’s James Webb Telescope captured new infrared images of Saturn’s rings, revealing temperature variations that could explain seasonal changes.”
  FAKE:
- “Scientists at MIT announced a handheld device that can turn any tap water directly into gasoline, ending the need for fuel stations.”
- “The World Health Organization secretly approved a plan to replace all vaccines with microchip implants by 2025, according to leaked documents.”
