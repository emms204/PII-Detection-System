"""
Evaluation script for Presidio PII detection
"""

from .detector import PresidioPiiDetector
from .utils import setup_logging, load_jsonl


def evaluate_presidio_detector():
    """
    Runs the PresidioPiiDetector on the test samples and prints the results.
    """
    logger = setup_logging("presidio_evaluation")
    
    logger.info("Initializing Presidio PII Detector...")
    try:
        detector = PresidioPiiDetector()
    except Exception as e:
        logger.error(f"Error initializing Presidio detector: {e}")
        logger.error("Please ensure you have installed presidio-analyzer and spacy model.")
        return
        
    logger.info("Starting evaluation on test samples...\n" + "="*40)
    
    try:
        test_samples = load_jsonl("pii_snippets.jsonl")
    except FileNotFoundError:
        logger.error("Evaluation file 'pii_snippets.jsonl' not found.")
        return

    total_found = 0
    total_ground_truth = sum(len(sample['pii']) for sample in test_samples)
    mismatch_count = 0
    
    for i, sample in enumerate(test_samples):
        logger.info(f"\n--- Sample {i} ---")
        logger.info(f"Text: {sample['text']}")
        
        pii_found = detector.detect(sample['text'])
        
        # Extract predicted PII types
        predicted_types = [pii['type'] for pii in pii_found]
        ground_truth_types = sample['pii']
        
        logger.info("PII Found:")
        if not pii_found:
            logger.info("  None")
            # Check if we missed any ground truth PII
            if ground_truth_types:
                logger.info(f"  MISSED: Ground truth had {ground_truth_types} but detected none")
                mismatch_count += 1
        else:
            for pii in pii_found:
                logger.info(f"  - Type: {pii['type']}, Value: '{pii['value']}', Confidence: {pii['confidence']:.2f}")
                # Simple accuracy check: does the detected type exist in the ground truth?
                if any(pii['type'] == pii_type for pii_type in sample['pii']):
                    total_found += 1
        
        # Check for mismatches between ground truth and predictions
        if not all(gt_type in predicted_types for gt_type in ground_truth_types):
            logger.info(f"  MISMATCH DETECTED!")
            logger.info(f"    Ground truth: {ground_truth_types}")
            logger.info(f"    Predicted:    {predicted_types}")
            mismatch_count += 1
            
            # Print to console for immediate visibility
            print(f"\n❌ MISMATCH in Sample {i}:")
            print(f"   Text: {sample['text']}")
            print(f"   Ground truth: {ground_truth_types}")
            print(f"   Predicted:    {predicted_types}")
        
        # This accuracy metric is a simple "found vs. ground truth" count and may not be perfect.
        print(f"Sample {i} running accuracy: {total_found}/{total_ground_truth}")
        logger.info(f"Sample {i} running accuracy: {total_found}/{total_ground_truth}")

    print(f"\nTotal Accuracy: {total_found}/{total_ground_truth}")
    print(f"Total Mismatches: {mismatch_count}")
    logger.info(f"Total Accuracy: {total_found}/{total_ground_truth}")
    logger.info(f"Total Mismatches: {mismatch_count}")
    logger.info("\n" + "="*40 + "\nEvaluation complete.")


if __name__ == '__main__':
    evaluate_presidio_detector()
