"""
Defines the PyTorch Dataset for loading and processing PII context data.
"""

import torch

class PiiTextDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset to handle tokenized text data.
    """
    def __init__(self, encodings, labels):
        """
        Args:
            encodings (transformers.tokenization_utils_base.BatchEncoding): 
                The tokenized inputs from the Hugging Face tokenizer.
            labels (list[int]): A list of integer labels.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Returns a single data point (input_ids, attention_mask, label).
        """
        # The encodings object itself is a dictionary-like object where keys
        # are 'input_ids', 'attention_mask', etc. We iterate through its keys
        # and get the tensor at the given index for each key.
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        # The number of samples is the length of the list of labels.
        return len(self.labels)
