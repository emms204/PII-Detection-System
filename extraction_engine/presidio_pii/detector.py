"""
Main Presidio PII detector class
"""

from typing import List, Dict, Any

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer

from .recognizers import get_custom_recognizers
from .utils import normalize_entity_type


class PresidioPiiDetector:
    """
    Detects various types of PII in a given text using Microsoft Presidio.
    """

    def __init__(self, 
                 context_similarity_factor: float = 0.4,
                 min_score_with_context: float = 0.5):
        """
        Initializes the Presidio AnalyzerEngine with custom recognizers and context enhancement.
        
        Args:
            context_similarity_factor: Boost factor when context words found
            min_score_with_context: Minimum score after context boost
        """
        # Get custom recognizers
        self.custom_recognizers = get_custom_recognizers()
        
        # Configure context enhancement engine
        context_enhancer = LemmaContextAwareEnhancer(
            context_similarity_factor=context_similarity_factor,
            min_score_with_context_similarity=min_score_with_context
        )
        
        # Initialize analyzer with custom recognizers and context enhancement
        self.analyzer = AnalyzerEngine(
            context_aware_enhancer=context_enhancer
        )
        
        # Add custom recognizers to the analyzer
        for recognizer in self.custom_recognizers:
            self.analyzer.registry.add_recognizer(recognizer)

        # Default entities to detect
        self.default_entities = [
            'PERSON', 'LOCATION', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 
            'CREDIT_CARD', 'DB_URI', 'API_KEY'
        ]

    def detect(self, text: str, entities: List[str] = None) -> List[Dict[str, Any]]:
        """
        Main detection method to find all PII types in the text.
        
        Args:
            text: Input text to analyze
            entities: List of entity types to detect (uses defaults if None)
            
        Returns:
            List of detected PII entities with type, value, position, and confidence
        """
        if entities is None:
            entities = self.default_entities
            
        analyzer_results = self.analyzer.analyze(
            text=text, 
            entities=entities, 
            language='en'
        )
        
        pii_results = []
        for result in analyzer_results:
            pii_results.append({
                "type": normalize_entity_type(result.entity_type),
                "value": text[result.start:result.end],
                "start": result.start,
                "end": result.end,
                "confidence": result.score
            })
            
        return pii_results

    def get_supported_entities(self) -> List[str]:
        """
        Get list of all supported entity types.
        
        Returns:
            List of supported entity type names
        """
        return [normalize_entity_type(entity) for entity in self.default_entities]

    def set_entities(self, entities: List[str]) -> None:
        """
        Set the default entities to detect.
        
        Args:
            entities: List of entity types to detect by default
        """
        self.default_entities = entities

    def add_custom_recognizer(self, recognizer) -> None:
        """
        Add a custom recognizer to the analyzer.
        
        Args:
            recognizer: Custom recognizer instance
        """
        self.analyzer.registry.add_recognizer(recognizer)
        self.custom_recognizers.append(recognizer)
