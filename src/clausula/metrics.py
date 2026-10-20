"""Evaluation helpers shared by every model (baseline, transformer, LLM)."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from clausula.labels import LABELS


def summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Macro-F1 (main metric), micro-F1 and binary F1 for "is this sentence unfair?"."""
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "unfair_f1": f1_score(y_true.any(axis=1), y_pred.any(axis=1), zero_division=0),
    }


def per_label(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Precision, recall, F1 and support for each category."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=LABELS)


def tune_thresholds(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Pick, for each category, the probability threshold that maximizes its F1.

    Must be fitted on the validation set, never on test.
    """
    grid = np.arange(0.05, 0.96, 0.05)
    thresholds = []
    for j in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, j], probs[:, j] >= t, zero_division=0) for t in grid]
        thresholds.append(grid[int(np.argmax(scores))])
    return np.array(thresholds)
