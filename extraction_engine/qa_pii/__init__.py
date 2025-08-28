"""
AnotiAI Question-Answering (QA) PII Detection Package

This package uses a pre-trained extractive QA model to find PII entities
by asking direct questions about the text.
"""

from .detector import QaPiiDetector

__version__ = "1.0.0"
__all__ = ["QaPiiDetector"]
