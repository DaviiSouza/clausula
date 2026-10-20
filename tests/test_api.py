import numpy as np
from fastapi.testclient import TestClient

from api.main import app, get_detector
from clausula.labels import LABELS


class FakeDetector:
    """Flags 'Arbitration' whenever the word appears. Avoids loading a real model in tests."""

    name = "fake"
    thresholds = np.full(len(LABELS), 0.5)

    def predict_proba(self, sentences):
        probs = np.zeros((len(sentences), len(LABELS)))
        for i, s in enumerate(sentences):
            if "arbitration" in s:
                probs[i, LABELS.index("Arbitration")] = 0.9
        return probs


app.dependency_overrides[get_detector] = lambda: FakeDetector()
client = TestClient(app)  # not used as a context manager, so the real model is never loaded


def test_health():
    assert client.get("/health").json() == {"status": "ok", "model": "fake"}


def test_analyze_flags_unfair_clause():
    text = "You must keep your password safe. Any dispute will be resolved by binding arbitration."
    body = client.post("/analyze", json={"text": text}).json()

    assert body["n_sentences"] == 2
    assert body["n_flagged"] == 1
    assert body["categories"] == {"Arbitration": 1}
    assert body["clauses"][0]["text"] == "Any dispute will be resolved by binding arbitration."
    assert body["clauses"][0]["labels"][0]["score"] == 0.9


def test_analyze_rejects_empty_text():
    assert client.post("/analyze", json={"text": ""}).status_code == 422


def test_pdf_endpoint_rejects_other_file_types():
    files = {"file": ("terms.txt", b"some text", "text/plain")}
    assert client.post("/analyze/pdf", files=files).status_code == 415
