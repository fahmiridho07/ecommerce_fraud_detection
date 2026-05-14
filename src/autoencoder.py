"""
Autoencoder module for the E-Commerce Fraud Detection project.

This module defines an Autoencoder neural network to learn the normal
distribution of the V-Features and extract Reconstruction Errors (MSE)
as a new feature for the downstream LightGBM model.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

# Allow GPU memory growth to prevent TF from hogging entire Kaggle GPU VRAM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model

# Minimize TensorFlow logging output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class VFeatureAutoencoder:
    """
    Autoencoder for dimensionality reduction and anomaly score extraction
    specifically on the V-Features set.
    """
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        """
        Initializes the Autoencoder architecture.
        
        Args:
            input_dim (int): Number of input features (V-features count).
            encoding_dim (int): Size of the bottleneck layer.
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """Constructs and compiles the Keras Autoencoder model."""
        # Clear previous sessions to prevent memory leaks in 8GB RAM environments
        tf.keras.backend.clear_session()
        
        input_layer = Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = Dense(128, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded) # Prevent overfitting
        bottleneck = Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(64, activation='relu')(bottleneck)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dropout(0.2)(decoded)
        output_layer = Dense(self.input_dim, activation='linear')(decoded)
        
        # Compile
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder

    def fit(self, X_train: pd.DataFrame, X_val: pd.DataFrame = None, 
            epochs: int = 50, batch_size: int = 512):
        """
        Trains the autoencoder to reconstruct the input data.
        
        Args:
            X_train (pd.DataFrame): Scaled training features.
            X_val (pd.DataFrame, optional): Scaled validation features for EarlyStopping.
            epochs (int): Maximum number of epochs.
            batch_size (int): Batch size (set to 512 to balance speed and memory).
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        validation_data = (X_val, X_val) if X_val is not None else None
        
        self.model.fit(
            x=X_train,
            y=X_train, # Target is the input itself
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0 # Suppress spammy output, EarlyStopping will print when it restores weights
        )

    def get_reconstruction_error(self, X: pd.DataFrame) -> pd.Series:
        """
        Calculates the Mean Squared Error (MSE) between the input and its reconstruction.
        This error serves as the anomaly score (higher error implies higher fraud likelihood).
        
        Args:
            X (pd.DataFrame): Scaled features to evaluate.
            
        Returns:
            pd.Series: The reconstruction error for each sample.
        """
        reconstructions = self.model.predict(X, batch_size=1024, verbose=0)
        
        # Calculate MSE per row (sample)
        mse = np.mean(np.power(X.values - reconstructions, 2), axis=1)
        
        return pd.Series(mse, index=X.index, name='AE_Reconstruction_Error')
