# Clausula

Detect potentially unfair clauses in Terms of Service. Paste a ToS (or upload a PDF) and get each suspicious clause highlighted, with its category, a confidence score and a plain-language explanation.

> 🚧 Work in progress.

## Data

[LexGLUE `unfair_tos`](https://huggingface.co/datasets/coastalcph/lex_glue): about 9.4k sentences from 50 real Terms of Service, labeled with 8 categories of potentially unfair clauses (CLAUDETTE corpus).

## Results (test set)

| Model | macro-F1 | ms/sentence | hardware |
|---|---|---|---|
| Always "fair" | 0.000 | – | – |
| TF-IDF + Logistic Regression | 0.765 | ~0.04 | CPU |
| distilroberta-base (fine-tuned) | 0.778 | ~1.5 | GPU |
| **legal-bert-base-uncased (fine-tuned, 3 seeds)** | **0.782 ± 0.006** | ~3 | GPU |

The transformer is stable across seeds and always beats the baseline, but a paired bootstrap shows the gap is not statistically significant on this small test set (95% CI of the difference: [−0.045, +0.075]). Details in [`notebooks/03_model_comparison.ipynb`](notebooks/03_model_comparison.ipynb).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

## Run

```bash
# 1. API (loads the model once at startup) → docs at http://localhost:8000/docs
uvicorn api.main:app --reload

# 2. Web app, in another terminal → http://localhost:8501
streamlit run app/streamlit_app.py

# Tests
pytest
```

Or with Docker (models are mounted from `./models`):

```bash
docker compose up --build
```

Set `CLAUSULA_BACKEND=baseline` to use the fast TF-IDF model instead of legal-bert.

### API

```bash
curl -X POST localhost:8000/analyze -H "Content-Type: application/json" \
  -d '{"text": "We may terminate your account at any time, for any reason."}'
```

`POST /analyze/pdf` accepts a PDF upload. The response lists each flagged clause with its categories, confidence scores and a plain-language explanation, plus an overall risk level (based on how many different kinds of unfair clause were found).

## Roadmap

- [x] EDA
- [x] Baseline: TF-IDF + Logistic Regression (macro-F1 0.765)
- [x] Fine-tuned transformer (legal-bert, macro-F1 0.782 ± 0.006)
- [ ] LLM zero-shot comparison
- [x] FastAPI `/analyze` endpoint
- [x] Streamlit app
- [x] Docker + CI
- [ ] Deploy (Hugging Face Spaces)
- [ ] Portuguese support (Brazilian consumer law, CDC art. 51)
