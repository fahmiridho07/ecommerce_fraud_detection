"""
Data Preprocessing module for the E-Commerce Fraud Detection project.

This module handles missing values, scaling, feature isolation,
and categorical encoding to prepare the data for Autoencoder and LightGBM models.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

class FraudDataPreprocessor:
    """
    A stateful preprocessor to handle transformations across cross-validation folds
    without causing data leakage.
    """
    def __init__(self):
        self.v_scaler = StandardScaler()
        self.v_columns = None
        self.categorical_cols = None
        self.categorical_dtypes = {}
        
    def _get_v_columns(self, X: pd.DataFrame) -> list:
        """Helper method to isolate V-features."""
        return [col for col in X.columns if col.startswith('V') and hasattr(X[col], 'dtype') and pd.api.types.is_numeric_dtype(X[col])]

    def fit_transform_v_features(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Isolates V-features from the training data, imputes missing values, 
        and fits/applies Z-score scaling. Essential for the Autoencoder.
        
        Args:
            X_train: Training feature dataframe.
            
        Returns:
            pd.DataFrame: Scaled V-features without NaNs.
        """
        self.v_columns = self._get_v_columns(X_train)
        
        # Isolate V-features
        X_v = X_train[self.v_columns].copy()
        
        # Autoencoders fail on NaNs. Impute with 0. 
        # Since we standardize immediately after, 0 becomes the mean-centered zero.
        X_v = X_v.fillna(0)
        
        # Fit and transform
        scaled_array = self.v_scaler.fit_transform(X_v)
        
        return pd.DataFrame(scaled_array, columns=self.v_columns, index=X_train.index)

    def transform_v_features(self, X_val: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms V-features of validation/test data using the PREVIOUSLY FITTED scaler.
        Prevents data leakage from validation/test fold into training.
        
        Args:
            X_val: Validation/Test feature dataframe.
            
        Returns:
            pd.DataFrame: Scaled V-features.
        """
        if not self.v_columns:
            raise ValueError("The scaler has not been fitted yet. Call `fit_transform_v_features` first.")
            
        X_v = X_val[self.v_columns].copy()
        X_v = X_v.fillna(0)
        
        # ONLY transform, DO NOT fit!
        scaled_array = self.v_scaler.transform(X_v)
        
        return pd.DataFrame(scaled_array, columns=self.v_columns, index=X_val.index)

    def fit_transform_lightgbm(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fits categorical states on training data & drops IDs."""
        X_out = X_train.copy()
        cols_to_drop = ['TransactionID', 'TransactionDT']
        X_out = X_out.drop(columns=[c for c in cols_to_drop if c in X_out.columns])
        
        self.categorical_cols = X_out.select_dtypes(include=['object']).columns
        for col in self.categorical_cols:
            X_out[col] = X_out[col].astype('category')
            # Save the categories dtype state strictly defined by Train set
            self.categorical_dtypes[col] = X_out[col].dtype
            
        return X_out

    def transform_lightgbm(self, X_val: pd.DataFrame) -> pd.DataFrame:
        """Transforms categorical columns identically to the Train set mapping."""
        if self.categorical_cols is None:
            raise ValueError("Call fit_transform_lightgbm first.")
        
        X_out = X_val.copy()
        cols_to_drop = ['TransactionID', 'TransactionDT']
        X_out = X_out.drop(columns=[c for c in cols_to_drop if c in X_out.columns])
        
        for col in self.categorical_cols:
            X_out[col] = X_out[col].astype(self.categorical_dtypes[col])
            
        return X_out
