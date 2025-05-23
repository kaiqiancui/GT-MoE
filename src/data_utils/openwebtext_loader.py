import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import numpy as np


class OpenWebTextDataset(Dataset):
    """
    Dataset for the OpenWebText dataset.
    
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
        Initialize OpenWebTextDataset.
        
        Args:
            split: Dataset split ('train' or 'validation')
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
        # OpenWebText only has a 'train' split, so we'll create a validation split
        if split == "train" or split == "test":
            dataset_split = "train"
        else:
            dataset_split = "train"  # We'll split this later for validation
            
        self.data = load_dataset(
            "openwebtext",
            split=dataset_split,
            streaming=streaming
        )
        
        # If not streaming, create train/validation/test splits
        if not streaming:
            if split != "train":
                # Use 2% for validation and 2% for test
                # First shuffle the dataset
                full_dataset = self.data.shuffle(seed=seed)
                
                # Get the size of the dataset
                dataset_size = len(full_dataset)
                
                # Calculate split sizes
                val_size = int(0.02 * dataset_size)
                test_size = int(0.02 * dataset_size)
                train_size = dataset_size - val_size - test_size
                
                # Split the dataset
                splits = full_dataset.train_test_split(test_size=val_size + test_size, seed=seed)
                train_dataset = splits['train']
                temp_dataset = splits['test']
                
                # Further split the test portion into validation and test
                temp_splits = temp_dataset.train_test_split(test_size=test_size / (val_size + test_size), seed=seed)
                val_dataset = temp_splits['train']
                test_dataset = temp_splits['test']
                
                # Assign the appropriate split
                if split == "validation":
                    self.data = val_dataset
                elif split == "test":
                    self.data = test_dataset
            
        # Shuffle if requested
        if shuffle and not streaming:
            self.data = self.data.shuffle(seed=seed)
        
        # Tokenize the dataset if tokenizer is provided
        if tokenizer is not None and not streaming:
            self.data = self.data.map(
                self._tokenize_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=["text"],
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


def create_openwebtext_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4,
    streaming: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for the OpenWebText dataset.
    
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
    train_dataset = OpenWebTextDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=max_length,
        streaming=streaming,
    )
    
    val_dataset = OpenWebTextDataset(
        split="validation",
        tokenizer=tokenizer,
        max_length=max_length,
        streaming=streaming,
    )
    
    test_dataset = OpenWebTextDataset(
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
