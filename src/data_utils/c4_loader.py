import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Iterator
import numpy as np
import os 

class C4Dataset(IterableDataset):
    """
    Dataset for the AllenAI C4 dataset in streaming mode.
    
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
        streaming: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        local_data_path: str = None,
        num_files_to_load: int = 13,
        file_pattern: str = "c4-train.{i:05d}-of-01024.json.gz",
        **kwargs
    ):
        """
        Initialize C4Dataset.
        
        Args:
            split: Dataset split ('train', 'validation')
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            return_tensors: Whether to return tensors or lists
            streaming: Whether to use streaming mode (should be True)
            shuffle: Whether to shuffle the dataset
            seed: Random seed
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
        self.streaming = streaming
        
        # Map the split names to the ones used in the C4 dataset
        if split == "validation":
            dataset_split = "validation"
        elif split == "test":
            # C4 doesn't have a test split, so we use validation for test as well
            dataset_split = "validation"
        else:
            dataset_split = "train"
        
        # 使用几个文件进行导入
        dataset_path = "/disks/sata2/kaiqian/.cache/huggingface/hub/datasets--allenai--c4/snapshots/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/"
        
        
        num_files_to_load = 13

        downloaded_files = [
            f"{dataset_path}c4-train.{i:05d}-of-01024.json.gz" 
            for i in range(num_files_to_load)
        ]

        # 使用本地文件路径
        dataset_path = local_data_path if local_data_path else "/disks/sata2/kaiqian/.cache/huggingface/hub/datasets--allenai--c4/snapshots/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/"
        num_files = num_files_to_load
        file_pat = file_pattern
        
        # 准备本地文件列表
        downloaded_files = [
            f"{dataset_path}{file_pat.format(i=i)}" 
            for i in range(num_files)
        ]
        
        # Load the dataset in streaming mode
        self.data = load_dataset(
            "json",
            data_files=downloaded_files,
            split=dataset_split, 
            streaming=streaming,
        )
        
        # Shuffle if requested
        if shuffle:
            self.data = self.data.shuffle(seed=seed, buffer_size=10000)
        
    def _tokenize_function(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize a single example.
        
        Args:
            example: Example with 'text' field
            
        Returns:
            Tokenized example
        """
        # Tokenize the text
        tokenized = self.tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt" if self.return_tensors else None,
        )
        
        # If tokenizer returns a batch dimension (which it does with return_tensors="pt"),
        # we need to squeeze it out since we're processing one example at a time
        if self.return_tensors:
            tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
        
        # Prepare labels for language modeling (same as input_ids)
        tokenized["labels"] = tokenized["input_ids"].clone() if self.return_tensors else tokenized["input_ids"].copy()
        
        # IMPORTANT: Set labels for padding tokens to -100 so they are ignored in loss calculation
        if self.tokenizer.pad_token_id is not None:
            if self.return_tensors:
                # Mask out padding tokens from labels (attention_mask == 0 means it's a padding token)
                tokenized["labels"][tokenized["attention_mask"] == 0] = -100
            else:
                # For non-tensor mode
                for i, mask_val in enumerate(tokenized["attention_mask"]):
                    if mask_val == 0:
                        tokenized["labels"][i] = -100
        
        return tokenized
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterator for the dataset.
        
        Returns:
            Iterator over tokenized examples
        """
        # Map the tokenization function over the streaming dataset
        for example in self.data:
            yield self._tokenize_function(example)


class C4MapDataset(Dataset):
    """
    Map-style dataset for the AllenAI C4 dataset (non-streaming mode).
    This is provided for compatibility with the existing code.
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
        local_data_path: str = None,
        num_files_to_load: int = 13,
        file_pattern: str = "c4-train.{i:05d}-of-01024.json.gz",
        **kwargs
    ):
        """
        Initialize C4MapDataset.
        
        Args:
            split: Dataset split ('train', 'validation')
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            return_tensors: Whether to return tensors or lists
            streaming: Whether to use streaming mode (ignored, always False)
            num_proc: Number of processes for preprocessing
            shuffle: Whether to shuffle the dataset
            seed: Random seed
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
        
        # Map the split names to the ones used in the C4 dataset
        if split == "validation":
            dataset_split = "validation"
        elif split == "test":
            # C4 doesn't have a test split, so we use validation for test as well
            dataset_split = "validation"
        else:
            dataset_split = "train"
        
        # 1. 定义本地文件的路径和数量
        # 使用传入的参数或默认值
        dataset_path = local_data_path if 'local_data_path' in kwargs else "/disks/sata2/kaiqian/.cache/huggingface/hub/datasets--allenai--c4/snapshots/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/"
        num_files = num_files_to_load if 'num_files_to_load' in kwargs else 13
        file_pat = file_pattern if 'file_pattern' in kwargs else "c4-train.{i:05d}-of-01024.json.gz"
        
        downloaded_files = [
            f"{dataset_path}{file_pat.format(i=i)}" 
            for i in range(num_files)
        ]

        # 2. 从本地 JSON 文件加载数据，而不是从 "allenai/c4" Hub 加载
        #    - 第一个参数改为 "json"
        #    - 使用 `data_files` 参数指定本地文件列表
        #    - streaming 必须为 False
        self.data = load_dataset(
            "json",
            data_files=downloaded_files,
            streaming=False,
            # 当从本地文件加载时，默认会创建一个 "train" split。
            # 如果你有单独的验证文件，可以在 data_files 中使用字典来指定，
            # 例如: data_files={'train': train_files, 'validation': val_files}
            # 这里我们简单地将所有文件都作为当前 split。
            split="train" 
        )
        
        # Shuffle if requested
        if shuffle:
            self.data = self.data.shuffle(seed=seed)
        
        # Tokenize the dataset if tokenizer is provided
        if tokenizer is not None:
            self.data = self.data.map(
                self._tokenize_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=["text", "timestamp", "url"],
                load_from_cache_file=False,  # Force reprocessing, ignore old cache
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
        if self.tokenizer.pad_token_id is not None:
            # Mask out padding tokens from labels (attention_mask == 0 means it's a padding token)
            tokenized["labels"][tokenized["attention_mask"] == 0] = -100
        
        return tokenized
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            Number of examples in the dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Tokenized example
        """
        # For non-streaming datasets, we can index directly
        example = self.data[idx]
        
        # If the dataset was not preprocessed, tokenize on-the-fly
        if self.tokenizer is not None and "input_ids" not in example:
            example = self._tokenize_function({"text": [example["text"]]})
            # Convert single-example batch to individual example
            example = {k: v[0] for k, v in example.items()}
        
        # Convert to tensors if needed
        if self.return_tensors and not isinstance(example["input_ids"], torch.Tensor):
            example = {k: torch.tensor(v) for k, v in example.items() if k in ["input_ids", "attention_mask", "labels"]}
        
        return example


def create_c4_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4,
    streaming: bool = True,
    processed_data_path: str = None,
    local_data_path: str = None,
    num_files_to_load: int = 13,
    file_pattern: str = "c4-train.{i:05d}-of-01024.json.gz",
    **kwargs  # Accept additional parameters from config
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for the C4 dataset.
    
    Args:
        tokenizer: Tokenizer for encoding text
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        streaming: Whether to use streaming mode
        processed_data_path: Path to preprocessed dataset (if available)
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    # Check if preprocessed data is available
    if processed_data_path and os.path.exists(processed_data_path):
        # Load preprocessed data from disk
        print(f"Loading pre-tokenized dataset from '{processed_data_path}'...")
        from datasets import load_from_disk
        
        try:
            tokenized_dataset = load_from_disk(processed_data_path)
            print("Dataset loaded successfully.")
            
            # Split dataset into train/val/test
            # First split into train and temp (99% train, 1% temp)
            split_dataset = tokenized_dataset.train_test_split(test_size=0.01, seed=42)
            train_dataset = split_dataset['train']
            
            # Then split temp into val and test (50% val, 50% test)
            temp_dataset = split_dataset['test']
            val_test_split = temp_dataset.train_test_split(test_size=0.5, seed=42)
            val_dataset = val_test_split['train']
            test_dataset = val_test_split['test']
            
            print(f"Dataset split into: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test examples")
            
            # Create PyTorch datasets
            from torch.utils.data import TensorDataset, Dataset
            
            # Create a wrapper dataset that converts tuples to dictionaries
            class DictionaryDataset(Dataset):
                def __init__(self, tensor_dataset):
                    self.tensor_dataset = tensor_dataset
                    self.keys = ["input_ids", "attention_mask", "labels"]
                
                def __len__(self):
                    return len(self.tensor_dataset)
                
                def __getitem__(self, idx):
                    # Get the tuple from the tensor dataset
                    tensors = self.tensor_dataset[idx]
                    # Convert to dictionary with the expected keys
                    return {key: tensor for key, tensor in zip(self.keys, tensors)}
            
            # Ensure labels are created for language modeling
            def prepare_lm_dataset(dataset):
                # Create labels (shift input_ids right)
                labels = dataset['input_ids'].clone()
                
                # Set padding tokens to -100 so they're ignored in loss calculation
                if tokenizer.pad_token_id is not None:
                    labels[dataset['attention_mask'] == 0] = -100
                
                # Create a TensorDataset and wrap it with our DictionaryDataset
                tensor_dataset = TensorDataset(dataset['input_ids'], dataset['attention_mask'], labels)
                return DictionaryDataset(tensor_dataset)
            
            # Use map-style datasets for preprocessed data
            streaming = False
            
            # Create TensorDatasets
            train_dataset = prepare_lm_dataset(train_dataset)
            val_dataset = prepare_lm_dataset(val_dataset)
            test_dataset = prepare_lm_dataset(test_dataset)
            
        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            print("Falling back to on-the-fly tokenization...")
            processed_data_path = None
    
    # If no preprocessed data or loading failed, use original approach
    if not processed_data_path or not os.path.exists(processed_data_path):
        # Create datasets using the original approach
        if streaming:
            train_dataset = C4Dataset(
                split="train",
                tokenizer=tokenizer,
                max_length=max_length,
                streaming=streaming,
            )
            
            val_dataset = C4Dataset(
                split="validation",
                tokenizer=tokenizer,
                max_length=max_length,
                streaming=streaming,
            )
            
            # C4 doesn't have a test split, so we use validation for test as well
            test_dataset = C4Dataset(
                split="validation",
                tokenizer=tokenizer,
                max_length=max_length,
                streaming=streaming,
            )
        else:
            train_dataset = C4MapDataset(
                split="train",
                tokenizer=tokenizer,
                max_length=max_length,
                streaming=streaming,
            )
            
            val_dataset = C4MapDataset(
                split="validation",
                tokenizer=tokenizer,
                max_length=max_length,
                streaming=streaming,
            )
            
            # C4 doesn't have a test split, so we use validation for test as well
            test_dataset = C4MapDataset(
                split="validation",
                tokenizer=tokenizer,
                max_length=max_length,
                streaming=streaming,
            )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not streaming,  # No need to shuffle if streaming, as it's done in the dataset
        num_workers=0 if streaming else num_workers,  # Must be 0 for IterableDataset
        pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if streaming else num_workers,  # Must be 0 for IterableDataset
        pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if streaming else num_workers,  # Must be 0 for IterableDataset
        pin_memory=True,
    )
    
    return train_dataloader, val_dataloader, test_dataloader
