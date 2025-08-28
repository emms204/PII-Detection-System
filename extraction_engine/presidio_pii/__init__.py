"""
AnotiAI Presidio PII Detection Package

A modular PII detection package built on Microsoft Presidio with custom recognizers
and enhanced detection capabilities.
"""

from .detector import PresidioPiiDetector
from .recognizers import EntropyBasedSecretRecognizer
from .utils import setup_logging, load_jsonl

__version__ = "1.0.0"
__all__ = [
    "PresidioPiiDetector", 
    "EntropyBasedSecretRecognizer",
    "setup_logging",
    "load_jsonl"
]
