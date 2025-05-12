"""API tests with a fake detector: fast, no model weights required."""
import numpy as np
import pytest

from sightector.inference import HarassmentDetector


class KeywordModel:
    name = "fake"

    def predict_proba(self, texts):
        return np.array([0.95 if "idiot" in t else 0.05 for t in texts])


@pytest.fixture()
def client():
    from server.app import create_app
    app = create_app(HarassmentDetector(KeywordModel(), threshold=0.5, kind="fake"))
    return app.test_client()


def test_health(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok" and body["model"] == "fake"


def test_analyze_detects_obfuscated_insult(client):
    body = client.post("/api/analyze", json={"text": "you are an iiid10t"}).get_json()
    assert body["harassment"] is True
    assert body["severity"] == "high"
    assert body["offensive_terms"] == ["idiot"]
    assert {"original": "iiid10t", "normalized": "idiot"} in body["deobfuscated"]
    assert body["highlights"][0]["word"] == "idiot"


def test_analyze_clean(client):
    body = client.post("/api/analyze", json={"text": "have a nice day"}).get_json()
    assert body["harassment"] is False and body["severity"] == "none"


@pytest.mark.parametrize("payload", [None, {}, {"text": ""}, {"text": 3}])
def test_analyze_rejects_bad_input(client, payload):
    assert client.post("/api/analyze", json=payload).status_code == 400


def test_batch(client):
    body = client.post("/api/analyze/batch", json={"texts": ["idiot", "hello"]}).get_json()
    assert [r["harassment"] for r in body] == [True, False]


def test_legacy_endpoint_is_compatible(client):
    body = client.post("/", json={"text": "idiot"}).get_json()
    assert body["harassment"] is True


def test_long_text_uses_windows():
    det = HarassmentDetector(KeywordModel(), threshold=0.5)
    text = "hello " * 500 + "idiot " + "nice " * 500
    assert det.analyze(text).harassment is True
