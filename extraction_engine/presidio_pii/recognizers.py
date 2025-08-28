"""
Custom recognizers for enhanced PII detection
"""

import re
import math
from typing import List

from presidio_analyzer import (
    PatternRecognizer,
    EntityRecognizer,
    Pattern,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import NlpArtifacts


class EntropyBasedSecretRecognizer(EntityRecognizer):
    """
    Custom recognizer for detecting high-entropy secrets using Shannon entropy analysis.
    This replaces the hardcoded API key prefix approach with a more robust method.
    """
    
    def __init__(self):
        super().__init__(
            supported_language="en",
            supported_entities=["API_KEY"],
            name="entropy_based_secret"
        )
        # Pattern for generic secrets (20+ characters, alphanumeric + underscore + hyphen)
        self.secret_regex = re.compile(r'\b[a-zA-Z0-9_\-]{20,}\b')

    def load(self) -> None:
        """No loading is required."""
        pass
    
    def _shannon_entropy(self, s):
        """Calculates the Shannon entropy for a string to detect randomness."""
        if not s:
            return 0
        entropy = 0
        for x in set(s):
            p_x = float(s.count(x)) / len(s)
            if p_x > 0:
                entropy += - p_x * math.log(p_x, 2)
        return entropy
    
    def analyze(self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts) -> List[RecognizerResult]:
        """
        Analyze text for high-entropy secrets.
        """
        results = []
        
        for match in self.secret_regex.finditer(text):
            token = match.group(0)
            entropy = self._shannon_entropy(token)
            
            # High entropy (>3.5) indicates randomness, likely a secret/API key
            if entropy > 3.5:
                # Lower base confidence since this is a weak pattern without context
                confidence = min(0.6, 0.3 + (entropy - 3.5) * 0.1)
                
                result = RecognizerResult(
                    entity_type="API_KEY",
                    start=match.start(),
                    end=match.end(),
                    score=confidence
                )
                results.append(result)
        
        return results


class DatabaseUriRecognizer(PatternRecognizer):
    """
    Recognizer for database connection URIs with context awareness.
    """
    
    def __init__(self):
        patterns = [
            Pattern(
                name="db_uri",
                regex=r'\b(?:sql|postgres|mongodb(?:\+srv)?)://[\w.-]+:[\w.-]+@[\w.-]+\b', 
                score=0.4  # Lower base confidence, will be boosted with context
            )
        ]
        
        super().__init__(
            supported_entity="DB_URI",
            supported_language="en",
            patterns=patterns,
            name="database_connection_string",
            context=[
                "db", "db_uri", "database", "connection", "uri", "url", 
                "connect", "host", "port", "mysql", "postgres", "mongodb"
            ]
        )


class PhoneNumberRecognizer(PatternRecognizer):
    """
    Enhanced phone number recognizer with context awareness.
    """
    
    def __init__(self):
        patterns = [
            Pattern(
                name="phone_number",
                regex=r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                score=0.3  # Lower base confidence, will be boosted with context
            ),
            Pattern(
                name="phone_number_international",
                regex=r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}',
                score=0.3  # Lower base confidence, will be boosted with context
            )
        ]
        
        super().__init__(
            supported_entity="PHONE_NUMBER",
            supported_language="en",
            patterns=patterns,
            name="phone_number_detector",
            context=[
                "phone", "telephone", "call", "dial", "number", "contact", 
                "mobile", "cell", "tel", "extension"
            ]
        )


def get_custom_recognizers():
    """
    Factory function to get all custom recognizers.
    """
    return [
        EntropyBasedSecretRecognizer(),
        DatabaseUriRecognizer(),
        PhoneNumberRecognizer()
    ]
