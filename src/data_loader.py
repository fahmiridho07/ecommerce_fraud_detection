"""
Data loading module for the E-Commerce Fraud Detection project.

This module provides environment-aware data execution, enforcing 
strict temporal ordering to prevent data leakage and dynamically 
managing data limits to prevent OOM errors.
"""

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import sys
from pathlib import Path

# Add project root to sys.path to allow direct execution
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import IS_KAGGLE, LOCAL_SAMPLE_SIZE, PROCESSED_DATA_DIR, RAW_DATA_DIR


def load_transaction_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads transaction data, ensuring strictly time-based sorting and 
    separating features from the target variable.
    
    Environment-aware execution logic:
    - Local (IS_KAGGLE = False): Restricts sample size using LOCAL_SAMPLE_SIZE 
      to prevent OOM errors on standard hardware (e.g., 8GB RAM).
    - Kaggle (IS_KAGGLE = True): Processes the full dataset.
    
    Returns:
        tuple: (X, y) where X contains the features and y contains the 'isFraud' target.
    """
    merged_data_path = PROCESSED_DATA_DIR / 'merged.parquet'
    
    # Load data
    df = pd.read_parquet(merged_data_path, engine='pyarrow')
    
    # 1. Guard against temporal leakage FIRST: strictly sort by TransactionDT
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    
    # 2. Limit based on environment AFTER sorting, not before
    if not IS_KAGGLE:
        # Local Environment: Restrict based on available memory threshold
        df = df.head(LOCAL_SAMPLE_SIZE)
    
    # Separate target and features
    y = df['isFraud']
    X = df.drop(columns=['isFraud'])
    
    return X, y


def get_cv_strategy() -> TimeSeriesSplit:
    """
    Configures and returns the project's cross-validation methodology.
    
    Returns:
        TimeSeriesSplit: A cv object configured for 5 splits without scaling.
    """
    return TimeSeriesSplit(n_splits=5)
