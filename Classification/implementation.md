# PII Detection System: A Phased Implementation Guide

## Overview

This document outlines a phased, "Context-First" approach for building a sophisticated and auditable Personally Identifiable Information (PII) detection and redaction system.

The core philosophy is to prioritize **understanding the context and intent** of a given text *before* attempting to extract any entities. This allows the system to intelligently distinguish between a benign **mention** of an entity (e.g., asking about a public figure) and a sensitive **disclosure** of private information (e.g., a user revealing their own or someone else's data).

This strategy ensures accuracy, reduces false positives, and provides a clear audit trail for every decision the system makes.

---

## Phase 1: Data Preparation & Corpus Development

**Objective:** To create a high-quality, three-class labeled dataset that will be the foundation for our context-aware model. The quality of this dataset will directly determine the system's performance.

### Key Tasks:

1.  **Define Classification Labels:**
    *   **`pii_disclosure`**: Texts where private, non-public information is being revealed.
        *   *Examples:* "My name is John Smith and my SSN is...", "Please send the package to Jane Doe at 123 Maple Lane.", "The suspect lives at 456 Oak Avenue."
    *   **`pii_inquiry_or_public_mention`**: Texts that mention entities that *could* be PII, but in a non-sensitive context like an inquiry, or a discussion about public figures or places.
        *   *Examples:* "I want to know more about Elon Musk.", "What is the address of the White House?", "Can you tell me about the history of Anderson Street?"
    *   **`no_pii`**: Texts that are definitively clean and do not contain any entities resembling PII.
        *   *Examples:* "This is a great product and the quality is excellent.", "The system should be rebooted after the installation is complete."

2.  **Source and Label Data:**
    *   Mine existing logs and files like `pii_snippets.jsonl` to find clear examples for the `pii_disclosure` class.
    *   Create or source new, distinct examples for the `pii_inquiry_or_public_mention` and `no_pii` classes. This may involve using public datasets (news articles, reviews) for the `no_pii` class.
    *   Establish strict labeling guidelines to ensure consistency.

3.  **Structure the Dataset:**
    *   Split the final labeled corpus into training, validation, and test sets (e.g., 70/15/15 split) to properly train and evaluate the model.

---

## Phase 2: Building the Contextual Classifier

**Objective:** To train a machine learning model that can accurately classify any given text into one of the three categories defined in Phase 1. This model will act as our intelligent "gatekeeper".

### Key Tasks:

1.  **Select Model Architecture:**
    *   We will use a **fine-tuned, self-hosted transformer model** (e.g., `DistilBERT`, `RoBERTa`). These models offer the best balance of contextual understanding, performance, and auditability without relying on external, "black-box" LLMs.

2.  **Develop the Training Pipeline:**
    *   Use a standard NLP framework like Hugging Face's `transformers` with `PyTorch` or `TensorFlow`.
    *   Set up a script to load the dataset from Phase 1, tokenize the text, and feed it to the model for fine-tuning.

3.  **Train and Evaluate:**
    *   Fine-tune the chosen transformer model on our labeled training data.
    *   Rigorously evaluate its performance on the held-out test set using metrics such as precision, recall, and F1-score for each of the three classes.
    *   Iterate on the data and model hyperparameters until a satisfactory level of accuracy is achieved.

---

## Phase 3: Developing the PII Extraction Engine

**Objective:** To build a robust and auditable toolkit for extracting the *specific* PII entities from a text, which will only be used *after* the text has been classified as `pii_disclosure`.

### Key Tasks:

1.  **Component 1: Rule-Based Extractor:**
    *   Develop a library of highly accurate regular expressions for structured PII (e.g., SSN, Email, Phone Number, IP Address).
    *   Implement logic checks, such as the Luhn algorithm for credit card numbers, to minimize false positives.
    *   Compile "gazetteers" (large dictionaries) of known entities like common names and locations to aid in detection.

2.  **Component 2: Question-Answering (QA) Extractor:**
    *   Select a transformer model fine-tuned for extractive question answering.
    *   Develop a set of standard questions to "ask" the model about the text, such as:
        *   "What is the person's name?"
        *   "What is the address?"
        *   "What is the Social Security Number?"
    *   This component is excellent for finding PII that doesn't follow a strict format.

---

## Phase 4: Integration and Final Pipeline

**Objective:** To assemble the trained classifier and the extraction engine into a single, cohesive, and intelligent PII detection workflow.

### Final Workflow:

1.  A new **Input Text** is received by the system.
2.  It is immediately passed to the **Contextual Classifier** (from Phase 2).
3.  The classifier predicts one of three labels:
    *   If the label is `no_pii` or `pii_inquiry_or_public_mention`, the process **STOPS**. The text is marked as safe.
    *   If the label is `pii_disclosure`, the text is passed to the **PII Extraction Engine** (from Phase 3).
4.  The Extraction Engine uses its rule-based and QA components to identify and list all specific PII entities within the text.
5.  The list of extracted entities is returned for redaction or other processing.

### Auditability:

This architecture provides a clear audit trail at every stage:
*   **Classification:** "This text was classified as `pii_disclosure` with 98% confidence." Explainability tools (SHAP) can further show which words influenced this decision.
*   **Extraction:** "The entity '987-65-4321' was extracted because it matched the `SSN_REGEX` rule," or "The entity 'Jane Doe' was extracted in response to the question 'What is the person's name?'."
