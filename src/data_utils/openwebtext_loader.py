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
            
        # Use a more reliable approach to load the dataset
        try:
            self.data = load_dataset(
                "openwebtext",
                trust_remote_code=True,
                split=dataset_split,
                streaming=False  # 明确设置为 False
            )
        except Exception as e:
            print(f"Error loading OpenWebText dataset: {e}")
            # Fallback to using the Pile dataset which contains similar web text
            self.data = load_dataset(
                "EleutherAI/pile", 
                name="all",
                split=dataset_split,
                streaming=False  # 明确设置为 False
            )
        
        # 创建 train/validation/test 拆分 (现在总是非流式模式)
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
        if shuffle:
            self.data = self.data.shuffle(seed=seed)
        
        # Tokenize the dataset if tokenizer is provided
        if tokenizer is not None:
            self.data = self.data.map(
                self._tokenize_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=["text"],
                load_from_cache_file=False, # <-- 关键修改：强制重新处理数据集，忽略旧缓存
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
        
        # IMPORTANT: Set labels for padding tokens to -100 so they are ignored in loss calculation
        # This assumes tokenizer.pad_token_id is correctly set (e.g., in tokenizer_utils.py)
        if self.tokenizer.pad_token_id is not None:
            # Mask out padding tokens from labels (attention_mask == 0 means it's a padding token)
            tokenized["labels"][tokenized["attention_mask"] == 0] = -100
        # For tokens beyond max_length that are truncated, their corresponding labels are naturally ignored
        # as they are not part of the input_ids.
        
        return tokenized
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            Number of examples in the dataset
        """
        # 在非流式模式下，self.data 是一个 Map-style Dataset，可以直接获取其长度
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Tokenized example
        """
        example = self.data[idx]
        
        # If the dataset was not preprocessed (e.g., tokenizer was None in __init__), tokenize on-the-fly
        # 这一段逻辑应该在 __init__ 中的 self.data.map 负责预处理后不再触发。
        # 但为了代码的健壮性，如果某些情况下 tokenizer 没传或没进行 map，可以保留。
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
