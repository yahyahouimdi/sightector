"""Application factory.

The detector is injected so tests can pass a lightweight fake, and the heavy
model is loaded exactly once per process.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # allow `python server/app.py` from anywhere
    sys.path.insert(0, str(ROOT))

from sightector.inference import HarassmentDetector  # noqa: E402

from .routes import api, web  # noqa: E402

MAX_TEXT_CHARS = 20_000


def create_app(detector: HarassmentDetector | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_TEXT_CHARS"] = MAX_TEXT_CHARS
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    app.extensions["detector"] = detector or HarassmentDetector.load()
    # The browser extension calls the API from a chrome-extension:// origin.
    CORS(app, resources={r"/*": {"origins": [r"chrome-extension://[a-z]+", r"moz-extension://[\w-]+",
                                              r"http://(localhost|127\.0\.0\.1)(:\d+)?"]}})
    app.register_blueprint(api)
    app.register_blueprint(web)
    logging.getLogger(__name__).info("Model loaded: %s", app.extensions["detector"].model.name)
    return app
