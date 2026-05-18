"""Project configuration for the IEEE-CIS fraud detection thesis experiments."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

KAGGLE_DATA_DIR = Path("/kaggle/input/competitions/ieee-fraud-detection")
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR = KAGGLE_DATA_DIR if KAGGLE_DATA_DIR.exists() else LOCAL_DATA_DIR

TRAIN_TRANSACTION_FILE = DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY_FILE = DATA_DIR / "train_identity.csv"
TEST_TRANSACTION_FILE = DATA_DIR / "test_transaction.csv"
TEST_IDENTITY_FILE = DATA_DIR / "test_identity.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
SPLIT_SUMMARY_FILE = OUTPUT_DIR / "split_summary.json"
BASELINE_OUTPUT_DIR = OUTPUT_DIR / "baseline_lgbm"
FEATURE_ENGINEERED_LGBM_OUTPUT_DIR = (
    OUTPUT_DIR / "baseline_lgbm_entity_time_amount_features"
)
AUTOENCODER_OUTPUT_DIR = OUTPUT_DIR / "autoencoder"
AUTOENCODER_ROBUST_OUTPUT_DIR = OUTPUT_DIR / "autoencoder_robust"
AUTOENCODER_ROBUST_LD64_OUTPUT_DIR = OUTPUT_DIR / "autoencoder_robust_ld64"
AUTOENCODER_ROBUST_LD128_OUTPUT_DIR = OUTPUT_DIR / "autoencoder_robust_ld128"
AE_LGBM_OUTPUT_DIR = OUTPUT_DIR / "ae_lgbm"
AE_LGBM_LD64_OUTPUT_DIR = OUTPUT_DIR / "ae_lgbm_ld64"
AE_LGBM_LD128_OUTPUT_DIR = OUTPUT_DIR / "ae_lgbm_ld128"
AE_AUGMENTED_LGBM_LD128_OUTPUT_DIR = OUTPUT_DIR / "ae_augmented_lgbm_ld128"
SCORE_ENSEMBLE_TUNED_OUTPUT_DIR = (
    OUTPUT_DIR / "score_ensemble_baseline_tuned_ae_lgbm_ld128_tuned"
)
RECON_ERROR_LGBM_ROBUST_RAW_OUTPUT_DIR = (
    OUTPUT_DIR / "baseline_lgbm_plus_ae_reconstruction_mse"
)
RECON_ERROR_LGBM_ROBUST_LOG1P_OUTPUT_DIR = (
    OUTPUT_DIR / "baseline_lgbm_plus_log1p_ae_reconstruction_mse"
)
RECON_ERROR_LGBM_ROBUST_RAW_LOG1P_OUTPUT_DIR = (
    OUTPUT_DIR / "baseline_lgbm_plus_raw_log1p_ae_reconstruction_mse"
)
AUTOENCODER_NORMAL_ONLY_LD128_OUTPUT_DIR = (
    OUTPUT_DIR / "autoencoder_normal_only_ld128"
)
RECON_ERROR_LGBM_NORMAL_ONLY_RAW_OUTPUT_DIR = (
    OUTPUT_DIR / "baseline_lgbm_plus_normal_only_ae_reconstruction_mse"
)
RECON_ERROR_LGBM_NORMAL_ONLY_LOG1P_OUTPUT_DIR = (
    OUTPUT_DIR / "baseline_lgbm_plus_log1p_normal_only_ae_reconstruction_mse"
)
RECON_ERROR_LGBM_NORMAL_ONLY_RAW_LOG1P_OUTPUT_DIR = (
    OUTPUT_DIR / "baseline_lgbm_plus_raw_log1p_normal_only_ae_reconstruction_mse"
)
OPTUNA_OUTPUT_DIR = OUTPUT_DIR / "optuna"
FINAL_COMPARISON_OUTPUT_DIR = OUTPUT_DIR / "final_comparison"
LATENT_DIM_ABLATION_FILE = FINAL_COMPARISON_OUTPUT_DIR / "latent_dim_ablation.csv"
AE_AUGMENTED_COMPARISON_FILE = FINAL_COMPARISON_OUTPUT_DIR / "ae_augmented_comparison.csv"
NEXT_CONTROLLED_EXPERIMENTS_COMPARISON_FILE = (
    FINAL_COMPARISON_OUTPUT_DIR / "next_controlled_experiments.csv"
)

RANDOM_SEED = 42

TARGET_COL = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

TRAIN_RATIO = 0.60
VALID_RATIO = 0.20
TEST_RATIO = 0.20

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
LATENT_DIM_ABLATION_DIMS = [64, 128]

OUTPUT_PATHS = {
    "baseline_lgbm": BASELINE_OUTPUT_DIR,
    "baseline_lgbm_entity_time_amount_features": (
        FEATURE_ENGINEERED_LGBM_OUTPUT_DIR
    ),
    "autoencoder": AUTOENCODER_OUTPUT_DIR,
    "autoencoder_robust": AUTOENCODER_ROBUST_OUTPUT_DIR,
    "autoencoder_robust_ld64": AUTOENCODER_ROBUST_LD64_OUTPUT_DIR,
    "autoencoder_robust_ld128": AUTOENCODER_ROBUST_LD128_OUTPUT_DIR,
    "ae_lgbm": AE_LGBM_OUTPUT_DIR,
    "ae_lgbm_ld64": AE_LGBM_LD64_OUTPUT_DIR,
    "ae_lgbm_ld128": AE_LGBM_LD128_OUTPUT_DIR,
    "ae_augmented_lgbm_ld128": AE_AUGMENTED_LGBM_LD128_OUTPUT_DIR,
    "score_ensemble_baseline_tuned_ae_lgbm_ld128_tuned": (
        SCORE_ENSEMBLE_TUNED_OUTPUT_DIR
    ),
    "baseline_lgbm_plus_ae_reconstruction_mse": (
        RECON_ERROR_LGBM_ROBUST_RAW_OUTPUT_DIR
    ),
    "baseline_lgbm_plus_log1p_ae_reconstruction_mse": (
        RECON_ERROR_LGBM_ROBUST_LOG1P_OUTPUT_DIR
    ),
    "baseline_lgbm_plus_raw_log1p_ae_reconstruction_mse": (
        RECON_ERROR_LGBM_ROBUST_RAW_LOG1P_OUTPUT_DIR
    ),
    "autoencoder_normal_only_ld128": AUTOENCODER_NORMAL_ONLY_LD128_OUTPUT_DIR,
    "baseline_lgbm_plus_normal_only_ae_reconstruction_mse": (
        RECON_ERROR_LGBM_NORMAL_ONLY_RAW_OUTPUT_DIR
    ),
    "baseline_lgbm_plus_log1p_normal_only_ae_reconstruction_mse": (
        RECON_ERROR_LGBM_NORMAL_ONLY_LOG1P_OUTPUT_DIR
    ),
    "baseline_lgbm_plus_raw_log1p_normal_only_ae_reconstruction_mse": (
        RECON_ERROR_LGBM_NORMAL_ONLY_RAW_LOG1P_OUTPUT_DIR
    ),
    "optuna": OPTUNA_OUTPUT_DIR,
    "final_comparison": FINAL_COMPARISON_OUTPUT_DIR,
}
