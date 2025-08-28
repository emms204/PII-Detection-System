"""
AnotiAI spaCy PII Detection Package

A modular PII detection package built on spaCy with custom validation
and enhanced pattern matching capabilities.
"""

from .detector import SpacyPiiDetector
from .validators import (
    validate_credit_card_luhn,
    validate_email_format, 
    validate_phone_number,
    validate_db_uri,
    validate_api_key_entropy,
    calculate_shannon_entropy
)
from .utils import setup_logging, load_jsonl

__version__ = "1.0.0"
__all__ = [
    "SpacyPiiDetector",
    "validate_credit_card_luhn",
    "validate_email_format", 
    "validate_phone_number",
    "validate_db_uri",
    "validate_api_key_entropy",
    "calculate_shannon_entropy",
    "setup_logging",
    "load_jsonl"
]
