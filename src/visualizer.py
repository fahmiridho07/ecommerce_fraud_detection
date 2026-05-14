"""
Visualization module for the E-Commerce Fraud Detection project.

This module provides plotting functions to evaluate model performance,
including PR-AUC curves, ROC curves, and Feature Importance logic.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, auc

def plot_pr_auc_curve(y_true: pd.Series, y_pred_proba: np.ndarray, title: str = "Precision-Recall Curve"):
    """
    Plots the Precision-Recall curve. Highly recommended for imbalanced datasets.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.4f})')
    
    # Calculate baseline (ratio of positive class)
    baseline = len(y_true[y_true == 1]) / len(y_true)
    plt.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--', label=f'Baseline ({baseline:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.show()

def plot_roc_curve(y_true: pd.Series, y_pred_proba: np.ndarray, title: str = "Receiver Operating Characteristic (ROC)"):
    """
    Plots the ROC curve for standard detection metric evaluation.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

def plot_feature_importance(model, feature_names: list, top_n: int = 20, title: str = "Top Feature Importances (LightGBM)"):
    """
    Plots the top N most important features from the trained LightGBM model.
    Useful for explaining whether the Autoencoder feature significantly contributed.
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute.")
        
    importances = model.feature_importances_
    
    # Create dataframe for sorting
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis')
    plt.title(title)
    plt.xlabel('Split/Gain Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
