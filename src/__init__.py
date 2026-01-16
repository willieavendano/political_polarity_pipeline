"""
Political Polarity Classification Pipeline

A reproducible NLP system for measuring political polarity in text corpora
with privacy-preserving aggregate analysis.
"""

__version__ = "1.0.0"
__author__ = "Your Organization"

from . import preprocessing
from . import features
from . import models
from . import evaluation
from . import interpretability
from . import privacy

__all__ = [
    "preprocessing",
    "features",
    "models",
    "evaluation",
    "interpretability",
    "privacy",
]
