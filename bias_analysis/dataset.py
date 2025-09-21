"""
Data loading utilities for Twitter poll bias analysis.

This module provides functions to load and parse JSONL files containing
Twitter poll data from multiple sources used in the 2020 US election analysis.
"""

from pathlib import Path
import json
import pandas as pd
from typing import Optional, Tuple

from loguru import logger
from tqdm import tqdm
import typer

from bias_analysis.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def load_jsonl_dataset(file_path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load JSONL (JSON Lines) dataset into pandas DataFrame.
    
    Handles malformed JSON lines gracefully by logging warnings and continuing
    to process valid lines. Progress is tracked with tqdm for large files.
    
    Args:
        file_path: Path to the JSONL file
        max_rows: Optional limit on number of rows to load (for testing/debugging)
        
    Returns:
        DataFrame with loaded data, empty DataFrame if no valid data found
    """
    data = []
    logger.info(f"Loading data from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {file_path.name}")):
            if max_rows and i >= max_rows:
                break
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {i+1} in {file_path}: {e}")
                continue
    
    if not data:
        logger.warning(f"No valid data found in {file_path}")
        return pd.DataFrame()

    logger.success(f"Loaded {len(data)} records from {file_path}")
    return pd.DataFrame(data)

def load_for_candidate_order(max_rows_per_file: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three primary Twitter poll datasets.
    
    Each dataset contains Twitter polls with candidate options and vote percentages
    that will be analyzed for position bias effects.
    
    Args:
        max_rows_per_file: Optional limit on rows per file (useful for development/testing)
        
    Returns:
        Tuple containing (decahose_df, vote_df, voting_df) DataFrames
        
    Raises:
        FileNotFoundError: If any required data file is missing
    """
    logger.info("Loading candidate order datasets...")
    
    # Define paths to the three core poll datasets
    decahose_input_path = RAW_DATA_DIR / "Decahose/polls.jsonl"
    vote_input_path = RAW_DATA_DIR / "vote/poll-vote-2020.jsonl"
    voting_input_path = RAW_DATA_DIR / "voting/poll-voting-2020.jsonl"
    
    # Validate all required files exist before attempting to load
    for path in [decahose_input_path, vote_input_path, voting_input_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")
    
    # Load each dataset with progress tracking
    decahose_df = load_jsonl_dataset(decahose_input_path, max_rows_per_file)
    vote_df = load_jsonl_dataset(vote_input_path, max_rows_per_file)
    voting_df = load_jsonl_dataset(voting_input_path, max_rows_per_file)
    
    logger.success("Data loading completed successfully!")
    
    return decahose_df, vote_df, voting_df


if __name__ == "__main__":
    logger.warning("dataset.py is not intended to be run directly. Use features.py instead.")
