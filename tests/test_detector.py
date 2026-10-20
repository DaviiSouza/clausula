import numpy as np

from clausula.detector import CascadeDetector
from clausula.labels import LABELS, grade


def test_grade_weights_kinds_of_clause_by_severity():
    assert grade({}) == "A"
    assert grade({"Contract by using": 3}) == "B"  # low impact, counted once however many clauses
    assert grade({"Arbitration": 1, "Unilateral termination": 1}) == "C"
    assert grade(dict.fromkeys(LABELS, 1)) == "E"


class FakeModel:
    def __init__(self, probs):
        self.probs, self.seen = probs, []
        self.thresholds = np.full(len(LABELS), 0.5)

    def predict_proba(self, sentences):
        self.seen += sentences
        return np.array([self.probs[s] for s in sentences])


def test_cascade_only_sends_suspicious_sentences_to_transformer():
    zero, one = np.zeros(len(LABELS)), np.eye(len(LABELS))[0]
    baseline = FakeModel({"fair": zero, "suspicious": one * 0.3})
    transformer = FakeModel({"suspicious": one * 0.9})

    probs = CascadeDetector(baseline, transformer).predict_proba(["fair", "suspicious"])

    assert transformer.seen == ["suspicious"]
    assert probs[0].sum() == 0 and probs[1][0] == 0.9
