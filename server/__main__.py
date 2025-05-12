"""Run the API: ``python -m server`` (dev) - use waitress/gunicorn in production."""
import os

from .app import create_app

if __name__ == "__main__":
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)),
                     debug=os.environ.get("FLASK_DEBUG") == "1")
