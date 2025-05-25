# src/preprocess_data.py

import os
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_and_save(config_path: str):
    """
    Loads a raw dataset from local files, tokenizes it based on a given configuration,
    and saves the processed dataset to disk for fast reuse.
    """
    # --- 1. Load Configuration ---
    logging.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    tokenizer_config = config['tokenizer']
    
    # Define key parameters from config
    local_data_path = data_config.get("local_data_path", "/disks/sata2/kaiqian/.cache/huggingface/hub/datasets--allenai--c4/snapshots/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/")
    num_files_to_load = data_config.get("num_files_to_load", 13)
    file_pattern = data_config.get("file_pattern", "c4-train.{i:05d}-of-01024.json.gz")
    tokenizer_name = tokenizer_config.get("tokenizer_name_or_path", "gpt2")
    output_dir = data_config.get("processed_data_path", "./processed_data/c4_tokenized")
    max_length = data_config.get("max_length", 512)
    num_proc = data_config.get("num_workers", 8) # Use num_workers for multiprocessing

    # --- 2. Load Tokenizer ---
    logging.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        logging.warning("Tokenizer does not have a pad token. Setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # --- 3. Load Raw Dataset from Local Files ---
    logging.info(f"Loading raw dataset from local files at: {local_data_path}")
    
    # Prepare the list of local files to load
    data_files = [
        f"{local_data_path}{file_pattern.format(i=i)}" 
        for i in range(num_files_to_load)
    ]
    
    logging.info(f"Loading {len(data_files)} files...")
    
    # Load dataset from local JSON files
    dataset = load_dataset(
        "json",
        data_files=data_files,
        streaming=False,
        split="train"
    )

    # --- 4. Define Tokenization Function ---
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None, # Let the map function handle tensor conversion
        )

    # --- 5. Apply Tokenization ---
    logging.info(f"Tokenizing dataset with {num_proc} processes... This may take a significant amount of time.")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names # Remove all original columns
    )

    # --- 6. Set Format for PyTorch ---
    logging.info("Setting dataset format to 'torch'")
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"]
    )

    # --- 7. Save Processed Dataset to Disk ---
    logging.info(f"Saving tokenized dataset to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(output_dir)

    logging.info(f"Preprocessing complete! Processed data saved at: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and save a dataset for training.")
    parser.add_argument(
        '--config_file', 
        type=str, 
        required=True,
        help='Path to the YAML configuration file (e.g., configs/rd_esi_small.yaml)'
    )
    args = parser.parse_args()
    preprocess_and_save(args.config_file)
