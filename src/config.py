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
UID_FEATURE_ENGINEERED_LGBM_OUTPUT_DIR = (
    OUTPUT_DIR / "baseline_lgbm_entity_time_amount_uid_features"
)
HISTORICAL_VELOCITY_LGBM_OUTPUT_DIR = (
    OUTPUT_DIR / "baseline_lgbm_entity_time_amount_historical_velocity_features"
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
FE_AE_CONTROLLED_OUTPUT_DIR = OUTPUT_DIR / "fe_ae_controlled_experiments"
FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR = (
    FE_AE_CONTROLLED_OUTPUT_DIR / "A_score_ensemble_fe_tuned_ae_tuned"
)
FE_RECON_ERROR_LGBM_OUTPUT_DIR = (
    FE_AE_CONTROLLED_OUTPUT_DIR / "B_fe_lgbm_reconstruction_mse_default"
)
FE_AE_AUGMENTED_LGBM_OUTPUT_DIR = (
    FE_AE_CONTROLLED_OUTPUT_DIR / "C_fe_lgbm_latent128_reconstruction_mse_default"
)
FE_AE_CONTROLLED_COMPARISON_FILE = FE_AE_CONTROLLED_OUTPUT_DIR / "comparison.csv"
BEHAVIORAL_CDV_AE_EXPERIMENT_OUTPUT_DIR = (
    OUTPUT_DIR / "behavioral_cdv_ae_experiment"
)
BEHAVIORAL_CDV_AUTOENCODER_LD128_OUTPUT_DIR = (
    BEHAVIORAL_CDV_AE_EXPERIMENT_OUTPUT_DIR / "autoencoder_cdv_ld128"
)
FE_CDV_RECON_ERROR_LGBM_OUTPUT_DIR = (
    BEHAVIORAL_CDV_AE_EXPERIMENT_OUTPUT_DIR
    / "A_fe_lgbm_cdv_reconstruction_mse_default"
)
BEHAVIORAL_CDV_AE_COMPARISON_FILE = (
    BEHAVIORAL_CDV_AE_EXPERIMENT_OUTPUT_DIR / "comparison.csv"
)
SPLIT_STRATEGY_APPENDIX_OUTPUT_DIR = OUTPUT_DIR / "split_strategy_appendix"
SPLIT_STRATEGY_APPENDIX_COMPARISON_FILE = (
    SPLIT_STRATEGY_APPENDIX_OUTPUT_DIR / "split_strategy_comparison.csv"
)
SPLIT_STRATEGY_APPENDIX_CV_FILE = (
    SPLIT_STRATEGY_APPENDIX_OUTPUT_DIR / "stratified_cv_summary.csv"
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
    "baseline_lgbm_entity_time_amount_uid_features": (
        UID_FEATURE_ENGINEERED_LGBM_OUTPUT_DIR
    ),
    "baseline_lgbm_entity_time_amount_historical_velocity_features": (
        HISTORICAL_VELOCITY_LGBM_OUTPUT_DIR
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
    "fe_ae_controlled_experiments": FE_AE_CONTROLLED_OUTPUT_DIR,
    "fe_ae_score_ensemble": FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR,
    "fe_reconstruction_error_lgbm": FE_RECON_ERROR_LGBM_OUTPUT_DIR,
    "fe_ae_augmented_lgbm": FE_AE_AUGMENTED_LGBM_OUTPUT_DIR,
    "behavioral_cdv_ae_experiment": BEHAVIORAL_CDV_AE_EXPERIMENT_OUTPUT_DIR,
    "behavioral_cdv_autoencoder_ld128": (
        BEHAVIORAL_CDV_AUTOENCODER_LD128_OUTPUT_DIR
    ),
    "fe_cdv_reconstruction_error_lgbm": FE_CDV_RECON_ERROR_LGBM_OUTPUT_DIR,
    "split_strategy_appendix": SPLIT_STRATEGY_APPENDIX_OUTPUT_DIR,
}
