"""Load a trained model and analyze a full Terms of Service document."""

import os
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from clausula.labels import EXPLANATIONS, LABELS, grade
from clausula.text import normalize, split_sentences

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
LOCAL_MODEL = MODELS_DIR / "legal-bert-base-uncased"
HUB_MODEL = "DaviiiiS/clausula-legal-bert"


class TransformerDetector:
    """Fine-tuned transformer (default: legal-bert). Thresholds tuned on validation."""

    name = "legal-bert"

    def __init__(self, model_dir: str | Path = LOCAL_MODEL, batch_size: int = 32):
        """`model_dir` is a local folder or a Hugging Face Hub repo id."""
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        local = Path(model_dir)
        thresholds = local / "thresholds.npy" if local.is_dir() else hf_hub_download(str(model_dir), "thresholds.npy")
        torch.set_num_threads(os.cpu_count() or 1)
        self.thresholds = np.load(thresholds)
        self.batch_size = batch_size

    def predict_proba(self, sentences: list[str]) -> np.ndarray:
        probs = np.zeros((len(sentences), len(LABELS)))
        # Batch sentences of similar length together: less padding, ~2x faster on CPU.
        order = np.argsort([len(s) for s in sentences])
        for i in range(0, len(sentences), self.batch_size):
            idx = order[i:i + self.batch_size]
            enc = self.tokenizer([sentences[j] for j in idx], truncation=True, max_length=256,
                                 padding=True, return_tensors="pt").to(self.device)
            with self._torch.inference_mode():
                probs[idx] = self._torch.sigmoid(self.model(**enc).logits).float().cpu().numpy()
        return probs


class BaselineDetector:
    """TF-IDF + logistic regression. Fast, CPU only."""

    name = "baseline"

    def __init__(self, path: str | Path | None = None):
        if path is None:
            local = MODELS_DIR / "baseline.joblib"
            path = local if local.exists() else _hub_file("baseline.joblib")
        saved = joblib.load(path)
        self.model, self.thresholds = saved["model"], saved["thresholds"]

    def predict_proba(self, sentences: list[str]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, len(LABELS)))
        return self.model.predict_proba(sentences)


class CascadeDetector:
    """Fast baseline screens every sentence; only "maybe unfair" ones go to the transformer.

    ~89% of ToS sentences are fair, so the transformer reads ~25% of the text: ~4x faster on CPU with
    the same macro-F1 (gate chosen on validation; see notebooks/04_cascade.ipynb).
    """

    name = "cascade (baseline → legal-bert)"
    GATE = 0.05

    def __init__(self, baseline: BaselineDetector, transformer: TransformerDetector):
        self.baseline, self.transformer = baseline, transformer
        self.thresholds = transformer.thresholds

    def predict_proba(self, sentences: list[str]) -> np.ndarray:
        probs = np.zeros((len(sentences), len(LABELS)))
        maybe_unfair = np.flatnonzero(self.baseline.predict_proba(sentences).max(axis=1) >= self.GATE)
        if len(maybe_unfair):
            probs[maybe_unfair] = self.transformer.predict_proba([sentences[i] for i in maybe_unfair])
        return probs


def _hub_file(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HUB_MODEL, filename)


def load_detector(backend: str | None = None):
    """Pick the backend from the argument or CLAUSULA_BACKEND: cascade (default), legal-bert or baseline."""
    backend = backend or os.getenv("CLAUSULA_BACKEND", "cascade")
    if backend == "baseline":
        return BaselineDetector()
    # Local model if it was trained on this machine, otherwise download it from the Hub.
    default = LOCAL_MODEL if LOCAL_MODEL.is_dir() else HUB_MODEL
    transformer = TransformerDetector(os.getenv("CLAUSULA_MODEL_DIR", default))
    if backend == "legal-bert":
        return transformer
    return CascadeDetector(BaselineDetector(), transformer)


def analyze(detector, text: str) -> dict:
    sentences = split_sentences(text)
    probs = detector.predict_proba([normalize(s) for s in sentences])

    clauses = []
    for sentence, p in zip(sentences, probs, strict=True):
        hits = [
            {"category": LABELS[j], "score": round(float(p[j]), 3), "explanation": EXPLANATIONS[LABELS[j]]}
            for j in np.flatnonzero(p >= detector.thresholds)
        ]
        if hits:
            clauses.append({"text": sentence, "labels": sorted(hits, key=lambda h: -h["score"])})

    counts = Counter(h["category"] for c in clauses for h in c["labels"])
    return {
        "model": detector.name,
        "n_sentences": len(sentences),
        "n_flagged": len(clauses),
        "grade": grade(counts),
        "categories": dict(counts.most_common()),
        "clauses": clauses,
    }
