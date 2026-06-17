"""Project configuration for the proposal-scope IEEE-CIS thesis pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

KAGGLE_DATA_DIR = Path("/kaggle/input/competitions/ieee-fraud-detection")
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR = KAGGLE_DATA_DIR if KAGGLE_DATA_DIR.exists() else LOCAL_DATA_DIR

TRAIN_TRANSACTION_FILE = DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY_FILE = DATA_DIR / "train_identity.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
SPLIT_SUMMARY_FILE = OUTPUT_DIR / "split_summary.json"
BASELINE_OUTPUT_DIR = OUTPUT_DIR / "baseline_lgbm"
AUTOENCODER_OUTPUT_DIR = OUTPUT_DIR / "autoencoder"
AUTOENCODER_ROBUST_OUTPUT_DIR = OUTPUT_DIR / "autoencoder_robust"
AUTOENCODER_ROBUST_LD128_OUTPUT_DIR = OUTPUT_DIR / "autoencoder_robust_ld128"
AE_LGBM_OUTPUT_DIR = OUTPUT_DIR / "ae_lgbm"
AE_LGBM_LD128_OUTPUT_DIR = OUTPUT_DIR / "ae_lgbm_ld128"
OPTUNA_OUTPUT_DIR = OUTPUT_DIR / "optuna"
FINAL_COMPARISON_OUTPUT_DIR = OUTPUT_DIR / "final_comparison"
STRATIFIED_RESET_OUTPUT_DIR = OUTPUT_DIR / "stratified_reset"
ALHARBI_STYLE_OUTPUT_DIR = STRATIFIED_RESET_OUTPUT_DIR / "alharbi_style_lgbm_default"

RANDOM_SEED = 42

TARGET_COL = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

TRAIN_RATIO = 0.60
VALID_RATIO = 0.20
TEST_RATIO = 0.20
DEFAULT_SPLIT_STRATEGY = "stratified_holdout"
SUPPORTED_SPLIT_STRATEGIES = ("chronological", "stratified_holdout")

# Set to an integer for quick local smoke tests. Keep as None for full runs.
SAMPLE_SIZE = None

MAIN_METRIC = "average_precision"
V_FEATURE_PATTERN = r"^V\d+$"

AE_LATENT_DIM = 32
AE_BATCH_SIZE = 1024
AE_MAX_EPOCHS = 100
AE_PATIENCE = 10
AE_LEARNING_RATE = 0.001
AE_USE_SCALED_CLIPPING = True
AE_CLIP_MIN = -10.0
AE_CLIP_MAX = 10.0
