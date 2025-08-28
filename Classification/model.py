"""
Defines the function to create the transformer model for sequence classification.
"""

from transformers import AutoModelForSequenceClassification, AutoConfig
import config

def create_model():
    """
    Creates and returns a pre-trained transformer model with a sequence 
    classification head.

    The number of labels is configured in the config.py file.

    Returns:
        transformers.PreTrainedModel: The instantiated model.
    """
    # Load the model configuration and update it with the number of labels
    model_config = AutoConfig.from_pretrained(
        config.MODEL_NAME, 
        num_labels=config.NUM_LABELS
    )
    
    # Load the pre-trained model with the updated configuration
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, 
        config=model_config
    )
    
    return model
