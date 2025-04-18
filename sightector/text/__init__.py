"""Text layer: lexicon, normalization and adversarial obfuscation."""
from .normalizer import find_offensive_terms, normalize, normalize_verbose

__all__ = ["find_offensive_terms", "normalize", "normalize_verbose"]
