"""
Script for hyperparameter tuning using Optuna.

This script uses the hyperparameter_search method from the Hugging Face Trainer
to find the best combination of hyperparameters for our model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TrainingArguments, Trainer
import optuna

import config
from dataset import PiiTextDataset
from model import create_model
from train import compute_metrics

def model_init():
    """
    This function is called by the Trainer to instantiate a new model for each trial.
    """
    return create_model()

def main():
    """
    Main function to run the hyperparameter tuning.
    """
    # Load and prepare data (same as in train.py)
    try:
        df = pd.read_csv(config.DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{config.DATA_PATH}'.")
        return

    df['label_id'] = df['label'].map(config.LABEL_MAP)
    train_val_df, _ = train_test_split(
        df, test_size=config.TEST_SPLIT_SIZE, random_state=42, stratify=df['label']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=config.VALIDATION_SPLIT_SIZE, random_state=42, stratify=train_val_df['label']
    )

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    train_texts = train_df['text'].tolist()
    train_labels = train_df['label_id'].astype(int).tolist()
    val_texts = val_df['text'].tolist()
    val_labels = val_df['label_id'].astype(int).tolist()

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=config.MAX_LEN)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=config.MAX_LEN)

    train_dataset = PiiTextDataset(train_encodings, train_labels)
    val_dataset = PiiTextDataset(val_encodings, val_labels)

    # Define training arguments (will be used as a template)
    training_args = TrainingArguments(
        output_dir='./tuning_results',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        disable_tqdm=True # Disables progress bars for cleaner logs
    )

    # Initialize Trainer
    trainer = Trainer(
        model_init=model_init, # Use model_init to get a new model for each trial
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Define the search space for Optuna
    def hp_space(trial: optuna.Trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        }

    print("Starting hyperparameter search...")
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=10, # Number of trials to run
        compute_objective=lambda metrics: metrics["eval_f1"],
    )

    print("\n--- Hyperparameter Search Complete ---")
    print(f"Best trial F1 score: {best_trial.objective:.4f}")
    print("Best hyperparameters:")
    for key, value in best_trial.hyperparameters.items():
        print(f"  {key}: {value}")
    print("--------------------------------------")

if __name__ == "__main__":
    main()
