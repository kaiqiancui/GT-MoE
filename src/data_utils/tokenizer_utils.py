from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Optional


def load_tokenizer(
    tokenizer_name_or_path: str = "gpt2",
    use_fast: bool = True,
    add_special_tokens: bool = True,
    padding_side: str = "right",
) -> PreTrainedTokenizerBase:
    """
    Load a tokenizer for the model.
    
    Args:
        tokenizer_name_or_path: Name or path of the tokenizer
        use_fast: Whether to use the fast tokenizer
        add_special_tokens: Whether to add special tokens
        padding_side: Side to pad on ('left' or 'right')
        
    Returns:
        Loaded tokenizer
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=use_fast,
    )
    
    # Set padding side
    tokenizer.padding_side = padding_side
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def get_tokenizer_info(tokenizer: PreTrainedTokenizerBase) -> dict:
    """
    Get information about the tokenizer.
    
    Args:
        tokenizer: The tokenizer
        
    Returns:
        Dictionary with tokenizer information
    """
    return {
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": tokenizer.model_max_length,
        "padding_token": tokenizer.pad_token,
        "padding_token_id": tokenizer.pad_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
    }
