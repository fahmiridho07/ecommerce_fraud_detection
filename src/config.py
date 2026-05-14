import os
from pathlib import Path

# Environment-Aware Logic
IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None

# Core Constants
SEED = 42
LOCAL_SAMPLE_SIZE = 5000

# Directory Setup
if IS_KAGGLE:
    # Typical Kaggle environment paths
    RAW_DATA_DIR = Path('/kaggle/input/ieee-fraud-detection')
    PROCESSED_DATA_DIR = Path('/kaggle/working/data/processed')
else:
    # Local environment paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
    PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

    # Ensure directories exist locally
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
