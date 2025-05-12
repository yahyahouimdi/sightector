"""Inference layer: the service object used by the API and the CLI."""
from .predictor import Analysis, HarassmentDetector

__all__ = ["Analysis", "HarassmentDetector"]
