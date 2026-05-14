"""
Trainer module for the E-Commerce Fraud Detection project.

This module handles hyperparameter tuning using Optuna (TPE)
and the training/evaluation of the LightGBM model, focusing on PR-AUC.
"""

import gc
import lightgbm as lgb
import optuna
import pandas as pd
from src.config import IS_KAGGLE
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score

class FraudLightGBMTrainer:
    """
    Handles Hyperparameter tuning (Optuna) and model training for LightGBM.
    Specifically designed to maximize PR-AUC for imbalanced fraud datasets.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_params = None
        self.model = None

    def optimize_optuna(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series, 
                        n_trials: int = 20) -> dict:
        """
        Runs Optuna Hyperparameter Optimization maximizing PR-AUC.
        
        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data for early stopping and metric evaluation.
            n_trials: Number of Optuna trials (keep low for local 8GB machines).
            
        Returns:
            dict: Best hyperparameters found by Optuna.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            # Define hyperparameter search space
            params = {
                'objective': 'binary',
                'metric': 'custom', # We will use custom PR-AUC evaluation
                'device_type': 'gpu' if IS_KAGGLE else 'cpu', # Enable GPU acceleration on Kaggle
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True), # L1 Regularization to prevent overfitting
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True), # L2 Regularization to prevent overfitting
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 50.0), # Critical for imbalanced data
                'random_state': self.random_state,
                'n_estimators': 500, # Large number with Early Stopping
                'verbose': -1
            }

            model = lgb.LGBMClassifier(**params)
            
            # Disable callbacks logging to keep terminal clean
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

            # Internal slice so we DONT LEAK the main X_val/y_val early-stopping/evaluation tests!
            # Since sorted temporally, last 20% of train becomes Optuna's test.
            trn_idx = int(len(X_train) * 0.8)
            X_opt_trn, y_opt_trn = X_train.iloc[:trn_idx], y_train.iloc[:trn_idx]
            X_opt_val, y_opt_val = X_train.iloc[trn_idx:], y_train.iloc[trn_idx:]

            model.fit(
                X_opt_trn, y_opt_trn,
                eval_set=[(X_opt_val, y_opt_val)],
                eval_metric=self._lgb_pr_auc_score,
                callbacks=callbacks
            )

            # Predict probabilities on the internal slice ONLY
            y_pred_proba = model.predict_proba(X_opt_val)[:, 1]
            pr_auc = average_precision_score(y_opt_val, y_pred_proba)
            
            # Explicit garbage collection to prevent memory ballooning during Optuna trials
            del model
            gc.collect()

            return pr_auc

        # Optimize (Maximize PR-AUC)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        
        # Merge best configurations (learning constraints)
        self.best_params.update({
            'objective': 'binary',
            'random_state': self.random_state,
            'n_estimators': 1000, 
            'verbose': -1
        })
        
        return self.best_params

    def train_final_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                          X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """
        Trains the final LightGBM model utilizing the best parameters.
        Automatically applies Early Stopping if validation data is provided.
        """
        if not self.best_params:
            raise ValueError("Run optimize_optuna() first or provide hyperparameters manually.")

        self.model = lgb.LGBMClassifier(**self.best_params)

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs['eval_set'] = [(X_val, y_val)]
            fit_kwargs['eval_metric'] = self._lgb_pr_auc_score
            fit_kwargs['callbacks'] = [lgb.early_stopping(stopping_rounds=100, verbose=100)]
            
        self.model.fit(X_train, y_train, **fit_kwargs)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates the trained model against true labels and returns a dictionary of metrics.
        Fokus utama pada PR-AUC (Average Precision).
        """
        if self.model is None:
            raise ValueError("Model is not trained yet.")
            
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'PR-AUC': average_precision_score(y_test, y_pred_proba),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
            'F1-Score': f1_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred)
        }
        return metrics

    @staticmethod
    def _lgb_pr_auc_score(y_true, y_pred):
        """Custom Evaluation Metric logic mapping for LightGBM with Zero-Positive safeguard."""
        if y_true.sum() == 0:
            return 'pr_auc', 0.0, True # Safeguard: Evaluasi 0 jika tak ada fraud di fold ini
        score = average_precision_score(y_true, y_pred)
        return 'pr_auc', score, True # True means larger is better
