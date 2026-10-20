"""Baseline model: TF-IDF features + one logistic regression per category."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


def build_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", OneVsRestClassifier(
            LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
        )),
    ])
