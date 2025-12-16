# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 12:58:27 2025

@author: sttrc
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# Fungsi untuk membuat fitur lagging (Time Series)
def create_time_series_features(df, target_col='Close', n_lags=5, forecast_horizon=1):
    """
    Membuat fitur 'lag' dari kolom target.
    
    n_lags: Jumlah hari masa lalu yang digunakan sebagai fitur (misalnya, 5 hari).
    forecast_horizon: Jumlah hari ke depan yang diprediksi (misalnya, 1 hari).
    """
    df_new = df.copy()
    
    # 1. Membuat Fitur Lag (X)
    for i in range(1, n_lags + 1):
        df_new[f'Lag_{i}'] = df_new[target_col].shift(i)
        
    # 2. Membuat Kolom Target (y)
    # Target adalah harga 'forecast_horizon' hari ke depan
    df_new['Target'] = df_new[target_col].shift(-forecast_horizon)
    
    # Menghapus baris dengan nilai NaN yang dibuat oleh lagging/shifting
    df_new = df_new.dropna()
    
    return df_new

# --- Memuat Data atau Membuat Data Dummy ---
file_path = 'nama_file_saham.csv'
try:
    df = pd.read_csv(file_path)
    # Pastikan data diurutkan berdasarkan tanggal
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').set_index('Date')
except FileNotFoundError:
    print("Gagal memuat file. Menggunakan data dummy.")
    # Membuat data dummy time series yang lebih realistis
    dates = pd.date_range(start='2020-01-01', periods=100)
    # Simulasi harga saham dengan sedikit tren dan noise
    base_price = np.arange(100) * 0.5 + 50
    close_price = base_price + np.random.randn(100) * 5
    df = pd.DataFrame({'Close': close_price}, index=dates)

# --- Penerapan Feature Engineering ---
N_LAGS = 5 # Gunakan 5 hari sebelumnya sebagai fitur
FORECAST_H = 1 # Prediksi 1 hari ke depan

df_ts = create_time_series_features(df, 'Close', N_LAGS, FORECAST_H)

print("\nData dengan Fitur Time Series (Lagging):")
print(df_ts.head(10))

# Definisikan Fitur (X) dan Target (y)
feature_cols = [col for col in df_ts.columns if col.startswith('Lag')]
X = df_ts[feature_cols]
y = df_ts['Target']

# 1. Definisikan Titik Pemisah (misalnya 80% data untuk Training)
train_size = int(len(df_ts) * 0.8)

# 2. Pembagian Data secara Sekuensial
X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

print(f"\nUkuran Data Latih (Train): {len(X_train)} sampel")
print(f"Ukuran Data Uji (Test): {len(X_test)} sampel")

# 3. Inisialisasi dan Latih Model Regresi Linear
model_ts = LinearRegression()
print("\nMelatih model Regresi Linear (Time Series)...")
model_ts.fit(X_train, y_train)

# 4. Melakukan Prediksi pada Data Uji
y_pred_ts = model_ts.predict(X_test)
print("Model selesai dilatih.")

# 5. Evaluasi Kinerja Model
mse_ts = mean_squared_error(y_test, y_pred_ts)
rmse_ts = np.sqrt(mse_ts)
r2_ts = r2_score(y_test, y_pred_ts)

print("\n=== Hasil Evaluasi Model Regresi Time Series ===")
print(f"Root Mean Squared Error (RMSE): {rmse_ts:.4f}")
print(f"R-squared (R²): {r2_ts:.4f}")

# 6. Visualisasi Hasil Prediksi Time Series
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test.values, label='Harga Aktual (y_test)', color='blue', linewidth=2)
plt.plot(y_test.index, y_pred_ts, label='Harga Prediksi (y_pred)', color='red', linestyle='--')


plt.title(f'Prediksi Harga Saham ({FORECAST_H} Hari ke Depan) Menggunakan Regresi Linear')
plt.xlabel('Tanggal')
plt.ylabel('Harga Penutupan (Close)')
plt.legend()
plt.grid(True)
plt.show()