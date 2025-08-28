"""
Main PII Detection Pipeline Orchestrator

This script implements the full, multi-stage PII detection workflow as
outlined in the implementation guide.

Workflow:
1.  Classify the input text for PII context ('pii_disclosure', 'pii_inquiry', 'no_pii').
2.  If the context is a 'pii_disclosure', proceed to extraction.
3.  Run multiple extraction engines (Presidio, spaCy, QA) in parallel.
4.  Consolidate and deduplicate the results from all engines.
5.  Return the final list of PII entities.
"""

import sys
import os
from typing import List, Dict, Any

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from Classification.predict import predict as classify_context
from extraction_engine.presidio_pii import PresidioPiiDetector
from extraction_engine.spacy_pii import SpacyPiiDetector
from extraction_engine.qa_pii import QaPiiDetector

class PiiDetectionPipeline:
    """
    Orchestrates the entire PII detection process.
    """

    def __init__(self):
        """
        Initializes and loads all necessary models and detectors.
        This can be slow, so it should be done once.
        """
        print("Initializing PII Detection Pipeline...")
        print("Loading extraction engines...")
        # The classifier model is loaded on-the-fly by the predict function.
        self.presidio_detector = PresidioPiiDetector()
        self.spacy_detector = SpacyPiiDetector()
        self.qa_detector = QaPiiDetector()
        print("Initialization complete.")

    def run(self, text: str) -> Dict[str, Any]:
        """
        Runs the full detection pipeline on a given text.

        Args:
            text (str): The input text to analyze.

        Returns:
            A dictionary containing the classification result and a list of found PII.
        """
        # 1. Classify the context
        print("\n--- Step 1: Classifying Context ---")
        classification, confidence = classify_context(text)

        if not classification:
            # This happens if the classifier model is not trained/found
            return {
                "classification": "ERROR",
                "confidence": 0.0,
                "pii_results": []
            }

        print(f"Result: Text classified as '{classification}' with {confidence:.2f} confidence.")

        # 2. Conditional Extraction
        if classification != "pii_disclosure":
            print("--- Step 2: Skipping Extraction ---")
            return {
                "classification": classification,
                "confidence": confidence,
                "pii_results": []
            }

        print("--- Step 2: Proceeding to Extraction ---")
        all_pii = []
        all_pii.extend(self.presidio_detector.detect(text))
        all_pii.extend(self.spacy_detector.detect(text))
        all_pii.extend(self.qa_detector.detect(text))

        # 3. Consolidate and Deduplicate Results
        print("--- Step 3: Consolidating Results ---")
        final_pii = self._consolidate_and_deduplicate(all_pii)
        print(f"Found {len(final_pii)} unique PII entities.")

        return {
            "classification": classification,
            "confidence": confidence,
            "pii_results": final_pii
        }

    def _consolidate_and_deduplicate(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes overlapping entities from multiple detector sources, keeping the one
        with the highest confidence score.
        """
        if not entities:
            return []

        # Sort by confidence score in descending order to prioritize high-confidence entities
        entities.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)

        unique_entities = []
        seen_ranges = set()

        for entity in entities:
            start, end = entity['start'], entity['end']
            # Check if the range of this entity overlaps with an already added entity
            if not any(start < seen_end and end > seen_start for seen_start, seen_end in seen_ranges):
                unique_entities.append(entity)
                seen_ranges.add((start, end))
        
        # Sort by start position for readability
        unique_entities.sort(key=lambda x: x['start'])
        return unique_entities


if __name__ == "__main__":
    # --- Initialize the Pipeline ---
    # Note: This may take a moment as it loads all the models.
    pipeline = PiiDetectionPipeline()

    # --- Define Sample Texts ---
    sample_disclosure = "The agent, John Doe, can be reached at john.doe@anotiai.com or 555-123-4567. His SSN is 987-65-4321."
    sample_inquiry = "Can you tell me who the CEO of AnotiAI is?"
    sample_no_pii = "The new software update will be released next week."

    # --- Run the Pipeline on Samples ---
    print("\n=======================================================")
    print(f"Running pipeline on DISCLOSURE sample...")
    disclosure_results = pipeline.run(sample_disclosure)
    print("\nFinal Output:")
    print(disclosure_results)
    print("=======================================================")

    print("\n=======================================================")
    print(f"Running pipeline on INQUIRY sample...")
    inquiry_results = pipeline.run(sample_inquiry)
    print("\nFinal Output:")
    print(inquiry_results)
    print("=======================================================")

    print("\n=======================================================")
    print(f"Running pipeline on NO PII sample...")
    no_pii_results = pipeline.run(sample_no_pii)
    print("\nFinal Output:")
    print(no_pii_results)
    print("=======================================================")
