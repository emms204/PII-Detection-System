"""
Main script to train the PII Context Classifier.

This script performs the following steps:
1. Loads the labeled dataset from the path specified in config.py.
2. Splits the dataset into training and validation sets.
3. Initializes the tokenizer and tokenizes the text data.
4. Creates PyTorch Datasets for training and validation.
5. Initializes the transformer model.
6. Defines training arguments and a compute_metrics function.
7. Initializes and runs the Hugging Face Trainer.
8. Saves the fine-tuned model to the path specified in config.py.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, TrainingArguments, Trainer

import config
from dataset import PiiTextDataset
from model import create_model

def compute_metrics(pred):
    """
    Computes accuracy, precision, recall, and F1-score from model predictions.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    """
    The main function to run the training pipeline.
    """
    # 1. Load Data
    try:
        df = pd.read_csv(config.DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{config.DATA_PATH}'.")
        print("Please update the DATA_PATH in config.py to point to your labeled CSV file.")
        return

    # Ensure required columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
        print("Error: The CSV file must contain 'text' and 'label' columns.")
        return

    # Map string labels to integers
    df['label_id'] = df['label'].map(config.LABEL_MAP)
    if df['label_id'].isnull().any():
        print("Error: One or more labels in the data are not in the LABEL_MAP in config.py.")
        return

    texts = df['text'].tolist()
    labels = df['label_id'].astype(int).tolist()

    # 2. Split Data into Train+Validation and Test
    # First, split off the test set.
    train_val_df, test_df = train_test_split(
        df, test_size=config.TEST_SPLIT_SIZE, random_state=42, stratify=df['label']
    )
    # Then, split the remainder into training and validation sets.
    train_df, val_df = train_test_split(
        train_val_df, test_size=config.VALIDATION_SPLIT_SIZE, random_state=42, stratify=train_val_df['label']
    )

    # Save the test set for later evaluation
    test_df.to_csv(config.TEST_DATA_PATH, index=False)
    print(f"Saved held-out test set to {config.TEST_DATA_PATH}")

    train_texts = train_df['text'].tolist()
    train_labels = train_df['label_id'].astype(int).tolist()
    val_texts = val_df['text'].tolist()
    val_labels = val_df['label_id'].astype(int).tolist()

    # 3. Tokenize
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=config.MAX_LEN)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=config.MAX_LEN)

    # 4. Create Datasets
    train_dataset = PiiTextDataset(train_encodings, train_labels)
    val_dataset = PiiTextDataset(val_encodings, val_labels)

    # 5. Initialize Model
    model = create_model()

    # 6. Define Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',                   # Output directory for checkpoints
        num_train_epochs=config.EPOCHS,              # Total number of training epochs
        learning_rate=config.LEARNING_RATE,        # Learning rate
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,  # Batch size for training
        per_device_eval_batch_size=config.VALID_BATCH_SIZE,    # Batch size for evaluation
        warmup_steps=500,                        # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,                       # Strength of weight decay
        logging_dir='./logs',                      # Directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",             # Evaluate at the end of each epoch
        save_strategy="epoch",                   # Save at the end of each epoch
        load_best_model_at_end=True,             # Load the best model when training is finished
        metric_for_best_model="f1",
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 8. Train
    print("Starting model training...")
    trainer.train()
    print("Training finished.")

    # 9. Save the best model
    print(f"Saving the best model to {config.MODEL_SAVE_PATH}...")
    trainer.save_model(config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(config.MODEL_SAVE_PATH)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
