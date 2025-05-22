import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import numpy as np


class C4TinyDataset(Dataset):
    """
    Dataset for the C4 tiny dataset.
    
    Attributes:
        data (datasets.Dataset): The loaded dataset
        tokenizer (PreTrainedTokenizerBase): Tokenizer for encoding text
        max_length (int): Maximum sequence length
        return_tensors (bool): Whether to return tensors or lists
    """
    
    def __init__(
        self,
        split: str = "train",
        tokenizer: PreTrainedTokenizerBase = None,
        max_length: int = 512,
        return_tensors: bool = True,
        streaming: bool = False,
        num_proc: int = 4,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize C4TinyDataset.
        
        Args:
            split: Dataset split ('train', 'validation', or 'test')
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            return_tensors: Whether to return tensors or lists
            streaming: Whether to use streaming mode
            num_proc: Number of processes for preprocessing
            shuffle: Whether to shuffle the dataset
            seed: Random seed
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
        
        # Load the dataset
        self.data = load_dataset(
            "allenai/c4", 
            "en", 
            split=split, 
            streaming=streaming
        )
        
        # Take a small subset for "tiny" version if not streaming
        if not streaming:
            if split == "train":
                self.data = self.data.select(range(100000))  # 100k examples for training
            else:
                self.data = self.data.select(range(10000))   # 10k examples for validation/test
        
        # Shuffle if requested
        if shuffle and not streaming:
            self.data = self.data.shuffle(seed=seed)
        
        # Tokenize the dataset if tokenizer is provided
        if tokenizer is not None and not streaming:
            self.data = self.data.map(
                self._tokenize_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=["text", "timestamp", "url"],
            )
        
    def _tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Tokenize a batch of examples.
        
        Args:
            examples: Batch of examples with 'text' field
            
        Returns:
            Tokenized examples
        """
        # Tokenize the texts
        tokenized = self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt" if self.return_tensors else None,
        )
        
        # Prepare labels for language modeling (same as input_ids)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            Number of examples in the dataset
        """
        if hasattr(self.data, "__len__"):
            return len(self.data)
        else:
            # For streaming datasets, return a large number
            return int(1e9)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Tokenized example
        """
        example = self.data[idx]
        
        # If the dataset is not preprocessed (streaming mode), tokenize on-the-fly
        if self.tokenizer is not None and "input_ids" not in example:
            example = self._tokenize_function({"text": [example["text"]]})
            # Convert single-example batch to individual example
            example = {k: v[0] for k, v in example.items()}
        
        # Convert to tensors if needed
        if self.return_tensors and not isinstance(example["input_ids"], torch.Tensor):
            example = {k: torch.tensor(v) for k, v in example.items() if k in ["input_ids", "attention_mask", "labels"]}
        
        return example


def create_c4_tiny_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4,
    streaming: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for the C4 tiny dataset.
    
    Args:
        tokenizer: Tokenizer for encoding text
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        streaming: Whether to use streaming mode
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    # Create datasets
    train_dataset = C4TinyDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=max_length,
        streaming=streaming,
    )
    
    val_dataset = C4TinyDataset(
        split="validation",
        tokenizer=tokenizer,
        max_length=max_length,
        streaming=streaming,
    )
    
    test_dataset = C4TinyDataset(
        split="test",
        tokenizer=tokenizer,
        max_length=max_length,
        streaming=streaming,
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not streaming,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_dataloader, val_dataloader, test_dataloader
