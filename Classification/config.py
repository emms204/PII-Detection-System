"""
Configuration settings and hyperparameters for the PII Context Classifier.
"""

# Model settings
MODEL_NAME = "distilbert-base-uncased"
MODEL_SAVE_PATH = "./p_ii_context_classifier_model"

# Dataset settings
# This should be the path to your labeled data file (e.g., a CSV).
# The file is expected to have two columns: "text" and "label".
DATA_PATH = "path/to/your/labeled_data.csv"
TEST_DATA_PATH = "./test_data.csv"  # Path to save the held-out test set


# Label mapping
# Ensures a consistent order and mapping for the labels.
LABEL_MAP = {
    "no_pii": 0,
    "pii_inquiry_or_public_mention": 1,
    "pii_disclosure": 2
}
# Create a reverse mapping for inference.
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)

# Tokenizer settings
MAX_LEN = 256  # Max sequence length

# Training hyperparameters
EPOCHS = 3
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8

# Dataset split
TEST_SPLIT_SIZE = 0.15
VALIDATION_SPLIT_SIZE = 0.15 # Percentage of the remaining data to use for validation
