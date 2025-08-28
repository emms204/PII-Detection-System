# Note on Optimization Techniques

This document clarifies the difference between two types of optimization relevant to our project: Hyperparameter Optimization and Parameter-Efficient Fine-Tuning (PEFT).

### 1. Hyperparameter Optimization (e.g., `optuna`, `hyperopt`)

- **What it is:** This is the process of finding the best *settings* or *hyperparameters* for our training process. These are high-level knobs we can turn to improve model performance.
- **Examples of Hyperparameters:** Learning Rate, Number of Training Epochs, Batch Size, Weight Decay.
- **Our Approach:** We will use **`optuna`**. It is a state-of-the-art library that integrates directly with the Hugging Face `Trainer`. It will automatically run multiple training "trials" with different hyperparameter combinations to find the one that yields the best F1 score on our validation set.
- **Why it's necessary:** Manually picking the learning rate or number of epochs is guesswork. A systematic search is crucial for maximizing model performance.

### 2. Parameter-Efficient Fine-Tuning (PEFT) (e.g., `LoRA`, `QLoRA`)

- **What it is:** This is a **method of training**, not a search for settings. Instead of fine-tuning all the millions of parameters in the transformer model (which is memory-intensive), PEFT techniques freeze the original model and only train a very small number of new parameters in "adapter" layers.
- **`LoRA` (Low-Rank Adaptation):** This is the most popular PEFT method. It injects small, trainable matrices into the model layers.
- **`QLoRA`:** This is an optimization of LoRA where the main model is first quantized to a lower precision (e.g., 4-bit) before the LoRA adapters are added. This saves a massive amount of memory, allowing huge models to be fine-tuned on a single GPU.

### Why Are We Using `optuna` and Not `LoRA`/`QLoRA` Right Now?

- **Necessity:** For a model of our current size (`distilbert-base-uncased`), a full fine-tuning (training all the layers) is perfectly feasible on most modern hardware. It does not consume an prohibitive amount of memory. Therefore, the extreme memory-saving benefits of `LoRA` or `QLoRA` are **not strictly necessary** at this moment.
- **Focus:** Our current priority is to find the best *hyperparameters* for a standard fine-tuning process. This is what `optuna` is for.
- **Future-Proofing:** If we were to scale up to a much larger model (e.g., a 7-billion parameter model like Llama 3 8B), a full fine-tune would be impossible on most systems. In that scenario, `QLoRA` would become **essential**. It is a powerful technique to keep in mind for future scaling.

**In summary:** We are using `optuna` to optimize our training *settings*. We are not using `LoRA` yet because our model is small enough to be trained directly, but it is the correct tool to use if we decide to scale up to a larger model architecture later.
