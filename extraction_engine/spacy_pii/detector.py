"""
Main spaCy PII detector class
"""

import re
import spacy
import phonenumbers
from typing import List, Dict, Any, Optional

from .validators import validate_pii_entity, calculate_shannon_entropy
from .utils import normalize_entity_type, deduplicate_entities


class SpacyPiiDetector:
    """
    Detects various types of PII in a given text using spaCy NER with custom patterns.
    """

    def __init__(self, model_name: str = "en_core_web_lg"):
        """
        Initialize the spaCy PII detector.
        
        Args:
            model_name: Name of the spaCy model to load
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise ImportError(f"SpaCy model '{model_name}' not found. Please run: python -m spacy download {model_name}")

        # Setup regex patterns
        self._setup_patterns()
        
        # Supported entity types
        self.supported_entities = [
            'name', 'email', 'credit_card', 'db_uri', 
            'api_key', 'phone', 'address'
        ]

    def _setup_patterns(self):
        """Setup regex patterns for detection."""
        # High-precision regexes
        self.credit_card_regex = re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b')
        self.db_uri_regex = re.compile(
            r'\b(?:mysql|postgres|mongodb(?:\+srv)?)://[\w.-]+:[\w.-]+@[\w.-]+\b', 
            re.IGNORECASE
        )
        self.email_regex = re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b')
        
        # API key patterns
        self.api_key_prefixes = [
            'sk_live_', 'pk_live_', 'rk_live_', 'sk_test_', 
            'pk_test_', 'rk_test_', 'AIza', 'ghp_', 'glpat-'
        ]
        self.generic_secret_regex = re.compile(r'\b[a-zA-Z0-9_\-]{20,}\b')

    def detect(self, text: str, validate: bool = True) -> List[Dict[str, Any]]:
        """
        Main detection method to find all PII types in the text.
        
        Args:
            text: Input text to analyze
            validate: Whether to run validation on detected entities
            
        Returns:
            List of detected PII entities with type, value, position, and confidence
        """
        pii_results = []
        
        # Use spaCy for NER-based detection first
        doc = self.nlp(text)
        
        # Detect Names and Addresses from NER
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                pii_results.append({
                    "type": "name",
                    "value": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 0.95  # High confidence for spaCy NER
                })
            elif ent.label_ in ["GPE", "LOC", "FAC"]:
                # Simple rule: if it has a number, it's more likely a specific address
                if any(char.isdigit() for char in ent.text):
                    pii_results.append({
                        "type": "address",
                        "value": ent.text,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 0.85
                    })

        # Regex-based detection
        pii_results.extend(self._detect_emails(text))
        pii_results.extend(self._detect_credit_cards(text, validate))
        pii_results.extend(self._detect_db_uris(text))
        pii_results.extend(self._detect_phone_numbers(text, validate))
        pii_results.extend(self._detect_api_keys(text))

        # Deduplicate overlapping results
        unique_pii = deduplicate_entities(pii_results)
        
        return unique_pii

    def _detect_emails(self, text: str) -> List[Dict[str, Any]]:
        """Detect email addresses."""
        results = []
        for match in self.email_regex.finditer(text):
            results.append({
                "type": "email",
                "value": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.99
            })
        return results

    def _detect_credit_cards(self, text: str, validate: bool = True) -> List[Dict[str, Any]]:
        """Detect credit card numbers with optional Luhn validation."""
        results = []
        for match in self.credit_card_regex.finditer(text):
            card_number = match.group(0)
            
            # Validate with Luhn algorithm if requested
            if validate and not validate_pii_entity('credit_card', card_number):
                continue
                
            results.append({
                "type": "credit_card",
                "value": card_number,
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.95 if validate else 0.75
            })
        return results

    def _detect_db_uris(self, text: str) -> List[Dict[str, Any]]:
        """Detect database URIs."""
        results = []
        for match in self.db_uri_regex.finditer(text):
            results.append({
                "type": "db_uri",
                "value": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.90
            })
        return results

    def _detect_phone_numbers(self, text: str, validate: bool = True) -> List[Dict[str, Any]]:
        """Detect phone numbers with optional validation."""
        results = []
        try:
            for match in phonenumbers.PhoneNumberMatcher(text, "US"):
                phone_number = phonenumbers.format_number(
                    match.number, 
                    phonenumbers.PhoneNumberFormat.E164
                )
                
                # Additional validation if requested
                confidence = 0.90
                if validate and not validate_pii_entity('phone', phone_number):
                    confidence = 0.60
                
                results.append({
                    "type": "phone",
                    "value": phone_number,
                    "start": match.start,
                    "end": match.end,
                    "confidence": confidence
                })
        except Exception:
            # Fallback if phonenumbers library fails
            pass
        return results

    def _detect_api_keys(self, text: str) -> List[Dict[str, Any]]:
        """Detect API keys using entropy and prefix analysis."""
        results = []
        for match in self.generic_secret_regex.finditer(text):
            token = match.group(0)
            
            # Check for known prefixes
            has_known_prefix = any(token.startswith(p) for p in self.api_key_prefixes)
            
            # Calculate entropy
            entropy = calculate_shannon_entropy(token)
            
            # Determine if it's likely an API key
            if entropy > 3.5 or has_known_prefix:
                confidence = 0.97 if has_known_prefix else min(0.75, 0.3 + (entropy - 3.5) * 0.1)
                
                results.append({
                    "type": "api_key",
                    "value": token,
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": confidence
                })
        return results

    def get_supported_entities(self) -> List[str]:
        """
        Get list of all supported entity types.
        
        Returns:
            List of supported entity type names
        """
        return self.supported_entities.copy()

    def set_supported_entities(self, entities: List[str]) -> None:
        """
        Set the supported entity types.
        
        Args:
            entities: List of entity types to support
        """
        self.supported_entities = entities
