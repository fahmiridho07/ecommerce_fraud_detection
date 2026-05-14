import sys
import gc
import warnings
from pathlib import Path

# Matikan warning agar notebook bersih
warnings.filterwarnings('ignore')

# Tambahkan root proyek ke sys.path agar direktori src/ dapat diakses
project_root = str(Path.cwd().resolve().parents[0])
if project_root not in sys.path:
    sys.path.append(project_root)

# Impor pipeline modules buatan kita
from src.data_loader import load_transaction_data, get_cv_strategy
from src.preprocessor import FraudDataPreprocessor
from src.autoencoder import VFeatureAutoencoder
from src.trainer import FraudLightGBMTrainer
from src.visualizer import plot_pr_auc_curve, plot_roc_curve, plot_feature_importance
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')

print('Environment setup complete.')

print("=== TAHAP 2: DATA LOADING ===")
X, y = load_transaction_data()
print(f"Total Fitur & Label: {X.shape}, {y.shape}")
print("Distribusi Fraud:")
print(y.value_counts(normalize=True) * 100)

print("=== TAHAP 3: TIME-SERIES CV ===")
ts_cv = get_cv_strategy()

# Mengambil Fold pertama sebagai contoh demonstrasi training Notebook
train_idx, val_idx = next(ts_cv.split(X))

X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()

# Bersihkan memori utama (Optimalisasi RAM 8GB)
del X, y 
gc.collect()

print(f"Train set shape: {X_train.shape}")
print(f"Val set shape: {X_val.shape}")

print("=== TAHAP 4: PREPROCESSING ===")
preprocessor = FraudDataPreprocessor()

# Z-Score Scaling V-features (hindari leakage menggunakan transform_v_features)
print(">> Scaling V-Features...")
X_train_v_scaled = preprocessor.fit_transform_v_features(X_train)
X_val_v_scaled = preprocessor.transform_v_features(X_val)

# Persiapan untuk LightGBM (mengubah tipe object menjadi category)
print(">> Persiapan LightGBM...")
X_train_lgb = preprocessor.fit_transform_lightgbm(X_train)
X_val_lgb = preprocessor.transform_lightgbm(X_val)

print("=== TAHAP 5: AUTOENCODER ===")
ae = VFeatureAutoencoder(input_dim=X_train_v_scaled.shape[1], encoding_dim=32)

print("Training Neural Network (5 Epochs sebagai representasi)...")
# Filter HANYA observasi "Wajar" (Fraud == 0) untuk melatih kewajaran Autoencoder!
normal_mask = (y_train == 0)
X_train_normal = X_train_v_scaled[normal_mask]

ae.fit(X_train_normal, X_val_v_scaled, epochs=5, batch_size=512)

# Mencetak MSE sebagai fitur anomali
X_train_lgb['AE_Reconstruction_Error'] = ae.get_reconstruction_error(X_train_v_scaled)
X_val_lgb['AE_Reconstruction_Error'] = ae.get_reconstruction_error(X_val_v_scaled)

del X_train_v_scaled, X_val_v_scaled, ae
gc.collect()

print("Berhasil menambahkan skor Anomali AE!")

print("=== TAHAP 6: OPTUNA TUNING ===")
trainer = FraudLightGBMTrainer(random_state=42)

# Cari hyperparameter terbaik
print("Pencarian parameter berjalan...")
best_params = trainer.optimize_optuna(X_train_lgb, y_train, X_val_lgb, y_val, n_trials=2)

print("\nParameter Terbaik:")
for k, v in best_params.items():
    print(f"{k}: {v}")

print("=== TAHAP 7: FINAL EVALUATION ===")
trainer.train_final_model(X_train_lgb, y_train, X_val_lgb, y_val)

metrics = trainer.evaluate(X_val_lgb, y_val)

print("\nMetrik Akhir:")
for metric, score in metrics.items():
    print(f" - {metric}: {score:.4f}")

# Visualisasi Evaluasi Model
y_pred_proba = trainer.model.predict_proba(X_val_lgb)[:, 1]

print("\nMenampilkan Grafik Evaluasi...")
plot_pr_auc_curve(y_val, y_pred_proba)
plot_roc_curve(y_val, y_pred_proba)

# Pemuatan Uji Variabel: Apakah sinyal AE_Reconstruction_Error di-approve LightGBM?
plot_feature_importance(trainer.model, feature_names=X_train_lgb.columns.tolist(), top_n=15)
