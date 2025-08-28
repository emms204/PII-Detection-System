Script to perform inference using the fine-tuned PII Context Classifier.

This script loads the saved model and tokenizer, and then uses them to
predict the class of a sample text.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import config

def predict(text: str):
    """
    Predicts the PII context class for a given text.

    Args:
        text (str): The input text to classify.

    Returns:
        tuple[str, float]: A tuple containing the predicted label and the 
                           confidence score.
    """
    # 1. Load the fine-tuned model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_SAVE_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_SAVE_PATH)
    except OSError:
        print(f"Error: Model not found at '{config.MODEL_SAVE_PATH}'.")
        print("Please run the train.py script first to train and save the model.")
        return None, None

    # 2. Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=config.MAX_LEN)

    # 3. Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # 4. Get prediction and confidence
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    predicted_class_id = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class_id].item()
    
    # 5. Map ID to label string
    predicted_label = config.ID_TO_LABEL[predicted_class_id]
    
    return predicted_label, confidence

def main():
    """
    Main function to demonstrate the prediction script.
    """
    # --- Example Texts ---
    text_disclosure = "My name is John Doe and I live at 123 Main Street."
    text_inquiry = "Can you tell me more about the history of the White House?"
    text_no_pii = "This is a test of the emergency broadcast system."

    print(f"Analyzing text: '{text_disclosure}'")
    label, score = predict(text_disclosure)
    if label:
        print(f" -> Predicted Label: {label}, Confidence: {score:.4f}\n")

    print(f"Analyzing text: '{text_inquiry}'")
    label, score = predict(text_inquiry)
    if label:
        print(f" -> Predicted Label: {label}, Confidence: {score:.4f}\n")

    print(f"Analyzing text: '{text_no_pii}'")
    label, score = predict(text_no_pii)
    if label:
        print(f" -> Predicted Label: {label}, Confidence: {score:.4f}\n")

if __name__ == "__main__":
    main()
