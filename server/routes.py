"""HTTP routes.

    GET  /                    dashboard with a live demo
    GET  /api/health          model information
    POST /api/analyze         {"text": "..."}            -> full explained analysis
    POST /api/analyze/batch   {"texts": ["...", ...]}    -> list of analyses (no explanations)
    POST /                    {"text": "..."}            -> v1-compatible {"harassment": bool, ...}
"""
from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template, request

from sightector import __version__

api = Blueprint("api", __name__, url_prefix="/api")
web = Blueprint("web", __name__)
MAX_BATCH = 100


def _detector():
    return current_app.extensions["detector"]


def _error(message: str, status: int):
    return jsonify({"error": message}), status


def _read_text():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return None, _error("Body must be JSON: {\"text\": \"...\"}", 400)
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, _error("Field 'text' must be a non-empty string", 400)
    if len(text) > current_app.config["MAX_TEXT_CHARS"]:
        return None, _error(f"Text longer than {current_app.config['MAX_TEXT_CHARS']} characters", 413)
    return text, None


@api.get("/health")
def health():
    det = _detector()
    return jsonify({"status": "ok", "version": __version__, "model": det.model.name,
                    "model_kind": det.kind, "threshold": det.threshold})


@api.post("/analyze")
def analyze():
    text, err = _read_text()
    if err:
        return err
    return jsonify(_detector().analyze(text).to_dict())


@api.post("/analyze/batch")
def analyze_batch():
    payload = request.get_json(silent=True) or {}
    texts = payload.get("texts")
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
        return _error("Field 'texts' must be a non-empty list of strings", 400)
    if len(texts) > MAX_BATCH:
        return _error(f"At most {MAX_BATCH} texts per request", 413)
    limit = current_app.config["MAX_TEXT_CHARS"]
    return jsonify([a.to_dict() for a in _detector().analyze_batch([t[:limit] for t in texts])])


@web.get("/")
def dashboard():
    return render_template("index.html", version=__version__, model=_detector().model.name)


@web.post("/")
def legacy_detect():
    """Kept so the v1 extension keeps working."""
    text, err = _read_text()
    if err:
        return err
    result = _detector().analyze(text, explain=False)
    return jsonify({"harassment": result.harassment, "score": result.score, "severity": result.severity})
