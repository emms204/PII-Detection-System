"""
Evaluation script for spaCy PII detection
"""

from .detector import SpacyPiiDetector
from .utils import setup_logging, load_jsonl


def evaluate_spacy_detector():
    """
    Runs the SpacyPiiDetector on the test samples and prints the results.
    """
    logger = setup_logging("spacy_evaluation")
    
    logger.info("Initializing spaCy PII Detector...")
    try:
        detector = SpacyPiiDetector()
    except ImportError as e:
        logger.error(f"Error: {e}")
        return
        
    logger.info("Starting evaluation on test samples...\n" + "="*40)
    
    try:
        test_samples = load_jsonl("pii_snippets.jsonl")
    except FileNotFoundError:
        logger.error("Evaluation file 'pii_snippets.jsonl' not found.")
        return

    gt_truth = sum(len(sample['pii']) for sample in test_samples)
    counter = 0
    
    for i, sample in enumerate(test_samples):
        logger.info(f"\n--- Sample {i} ---")
        logger.info(f"Text: {sample['text']}")
        pii_found = detector.detect(sample['text'])
        logger.info("PII Found:")
        if not pii_found:
            logger.info("  None")
        else:
            for pii in pii_found:
                # Check if the detected PII type exists in the sample's pii list
                if any(pii['type'].lower() == pii_type.lower() for pii_type in sample['pii']):
                    counter += 1
                logger.info(f"  - Type: {pii['type']}, Value: '{pii['value']}', Confidence: {pii['confidence']:.2f}")
        
        # Keep print statements for immediate console feedback
        print(f"{i} Accuracy: {counter}/{gt_truth}")
        logger.info(f"Sample {i} Accuracy: {counter}/{gt_truth}")

    print(f"Total Accuracy: {counter}/{gt_truth}")
    logger.info(f"Total Accuracy: {counter}/{gt_truth}")
    logger.info("\n" + "="*40 + "\nEvaluation complete.")


if __name__ == '__main__':
    evaluate_spacy_detector()
