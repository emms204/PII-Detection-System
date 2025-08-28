"""
Script to evaluate the fine-tuned model on the held-out test set.

This script performs the following steps:
1. Loads the held-out test set from the path specified in config.py.
2. Loads the best fine-tuned model and tokenizer.
3. Creates a PyTorch Dataset for the test data.
4. Initializes the Trainer.
5. Runs the evaluation and prints the performance metrics.
6. Generates and saves a confusion matrix.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, Trainer, TrainingArguments

import config
from dataset import PiiTextDataset
from model import create_model
from train import compute_metrics

def main():
    """
    Main function to run the evaluation.
    """
    # 1. Load Test Data
    try:
        test_df = pd.read_csv(config.TEST_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Test data file not found at '{config.TEST_DATA_PATH}'.")
        print("Please run train.py first to generate the test set.")
        return

    # Map labels to IDs
    test_df['label_id'] = test_df['label'].map(config.LABEL_MAP)
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label_id'].astype(int).tolist()

    # 2. Load Model and Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_SAVE_PATH)
        model = create_model() # Re-create model structure
        model.load_state_dict(torch.load(f"{config.MODEL_SAVE_PATH}/pytorch_model.bin"))
    except OSError:
        print(f"Error: Model not found at '{config.MODEL_SAVE_PATH}'.")
        print("Please run train.py first to train and save a model.")
        return

    # 3. Create Test Dataset
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=config.MAX_LEN)
    test_dataset = PiiTextDataset(test_encodings, test_labels)

    # 4. Initialize Trainer
    # We only need minimal TrainingArguments for evaluation
    eval_args = TrainingArguments(
        output_dir='./eval_results',
        per_device_eval_batch_size=config.VALID_BATCH_SIZE,
        do_train=False,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # 5. Evaluate
    print("Evaluating model on the held-out test set...")
    results = trainer.evaluate()

    print("\n--- Test Set Evaluation Results ---")
    for key, value in results.items():
        print(f"{key.replace('eval_', '').capitalize():<10}: {value:.4f}")
    print("-------------------------------------")

    # 6. Generate and Save Confusion Matrix
    print("\nGenerating confusion matrix...")
    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(axis=1)
    cm = confusion_matrix(test_labels, predicted_labels)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=config.LABEL_MAP.keys(), 
                yticklabels=config.LABEL_MAP.keys(),
                cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved to confusion_matrix.png")

if __name__ == "__main__":
    import torch
    main()
