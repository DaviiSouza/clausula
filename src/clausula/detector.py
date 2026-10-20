"""Load a trained model and analyze a full Terms of Service document."""

import os
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from clausula.labels import EXPLANATIONS, LABELS
from clausula.text import normalize, split_sentences

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class TransformerDetector:
    """Fine-tuned transformer (default: legal-bert). Thresholds tuned on validation."""

    name = "legal-bert"

    def __init__(self, model_dir: str | Path = MODELS_DIR / "legal-bert-base-uncased", batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.thresholds = np.load(Path(model_dir) / "thresholds.npy")
        self.batch_size = batch_size

    def predict_proba(self, sentences: list[str]) -> np.ndarray:
        probs = []
        for i in range(0, len(sentences), self.batch_size):
            enc = self.tokenizer(sentences[i:i + self.batch_size], truncation=True, max_length=256,
                                 padding=True, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                probs.append(self._torch.sigmoid(self.model(**enc).logits).float().cpu().numpy())
        return np.concatenate(probs) if probs else np.zeros((0, len(LABELS)))


class BaselineDetector:
    """TF-IDF + logistic regression. Fast, CPU only."""

    name = "baseline"

    def __init__(self, path: str | Path = MODELS_DIR / "baseline.joblib"):
        saved = joblib.load(path)
        self.model, self.thresholds = saved["model"], saved["thresholds"]

    def predict_proba(self, sentences: list[str]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, len(LABELS)))
        return self.model.predict_proba(sentences)


def load_detector(backend: str | None = None):
    """Pick the backend from the argument or the CLAUSULA_BACKEND env var (default: legal-bert)."""
    backend = backend or os.getenv("CLAUSULA_BACKEND", "legal-bert")
    if backend == "baseline":
        return BaselineDetector()
    return TransformerDetector(os.getenv("CLAUSULA_MODEL_DIR", MODELS_DIR / "legal-bert-base-uncased"))


def risk_level(n_categories: int) -> str:
    """Simple, explainable rule: how many *different* kinds of unfair clause were found."""
    if n_categories <= 1:
        return "low"
    if n_categories <= 4:
        return "medium"
    return "high"


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
        "risk_level": risk_level(len(counts)),
        "categories": dict(counts.most_common()),
        "clauses": clauses,
    }
