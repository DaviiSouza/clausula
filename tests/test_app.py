"""Drive the Streamlit app headlessly, with the API replaced by an in-process fake."""

import re

import pytest
import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api.main import app as api
from api.main import get_detector
from tests.test_api import FakeDetector


@pytest.fixture
def fake_api(monkeypatch):
    api.dependency_overrides[get_detector] = lambda: FakeDetector()
    client = TestClient(api)
    monkeypatch.setattr(requests, "post", lambda url, timeout=None, **kw: client.post(url.split("8000")[-1], **kw))


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, at.exception


def test_example_flow(fake_api):
    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=30).run()
    assert not at.exception

    click(at, "Try an example")
    assert at.text_area[0].value.startswith("Welcome to StreamBox")

    click(at, "Analyze")
    page = " ".join(m.value for m in at.markdown)
    assert re.search(r'grade-letter[^>]*>B<', page)  # the fake only flags arbitration (high impact → B)
    assert "You give up your right to go to court" in page
    assert not at.error
