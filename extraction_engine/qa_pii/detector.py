
"""
Main QA PII detector class.
"""

from typing import List, Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

from .questions import PII_QUESTIONS

class QaPiiDetector:
    """
    Detects PII by asking a series of questions to a fine-tuned QA model.
    """

    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        """
        Initializes the QA pipeline with a pre-trained model.

        Args:
            model_name (str): The name of the QA model to load from Hugging Face.
        """
        try:
            # Using a specific model and tokenizer ensures consistency
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.qa_pipeline = pipeline(
                "question-answering", 
                model=model, 
                tokenizer=tokenizer
            )
            print(f"QA model '{model_name}' loaded successfully.")
        except Exception as e:
            raise ImportError(f"Failed to load QA model '{model_name}'. Please check your connection and dependencies. Error: {e}")

        self.questions_map = PII_QUESTIONS

    def detect(self, text: str, confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Main detection method to find PII by asking questions.

        Args:
            text (str): The input text to analyze.
            confidence_threshold (float): The minimum confidence score to accept an answer.

        Returns:
            List[Dict[str, Any]]: A list of detected PII entities.
        """
        pii_results = []

        for pii_type, questions in self.questions_map.items():
            for question in questions:
                try:
                    result = self.qa_pipeline(question=question, context=text)
                    
                    # If the model's confidence is high enough, add it to our results
                    if result['score'] >= confidence_threshold:
                        pii_results.append({
                            "type": pii_type,
                            "value": result['answer'],
                            "start": result['start'],
                            "end": result['end'],
                            "confidence": result['score'],
                            "audited_by": f"question: '{question}'" # Add audit trail
                        })
                except Exception:
                    # The pipeline can sometimes fail on certain inputs
                    continue
        
        # It's possible to get duplicate answers from different questions, so we should deduplicate.
        return self._deduplicate_entities(pii_results)

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes overlapping entities, keeping the one with the highest confidence.
        """
        if not entities:
            return []

        # Sort by confidence score in descending order
        entities.sort(key=lambda x: x['confidence'], reverse=True)

        unique_entities = []
        seen_ranges = set()

        for entity in entities:
            # Check if the range of this entity overlaps with an already added entity
            if not any(entity['start'] < end and entity['end'] > start for start, end in seen_ranges):
                unique_entities.append(entity)
                seen_ranges.add((entity['start'], entity['end']))
        
        return unique_entities


# Example usage:
if __name__ == '__main__':
    detector = QaPiiDetector()
    sample_text = "My name is John Doe and my email is john.doe@example.com. Please send mail to 123 Main St."
    
    print(f"\nAnalyzing text: '{sample_text}'")
    pii_found = detector.detect(sample_text, confidence_threshold=0.5)

    print("\n--- PII Found ---")
    if not pii_found:
        print("None")
    else:
        for pii in pii_found:
            print(f"- Type: {pii['type']}, Value: '{pii['value']}', Confidence: {pii['confidence']:.4f}")
            print(f"  (Found by asking: {pii['audited_by']})")
    print("-----------------")
