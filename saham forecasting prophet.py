# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 13:02:20 2025

@author: sttrc
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import matplotlib.pyplot as plt

# --- 1. Memuat Data atau Membuat Data Dummy ---
file_path = 'nama_file_saham.csv'
try:
    df = pd.read_csv(file_path)
    # Pastikan data memiliki kolom tanggal dan harga penutupan
    if 'Date' in df.columns and 'Close' in df.columns:
        df = df[['Date', 'Close']]
    else:
        raise ValueError("File harus memiliki kolom 'Date' dan 'Close'")
except FileNotFoundError:
    print("Gagal memuat file. Menggunakan data dummy.")
    # Membuat data dummy time series
    dates = pd.date_range(start='2020-01-01', periods=200, freq='B') # B = Business days
    # Simulasi harga saham dengan tren dan musiman (sedikit lebih kompleks)
    base_price = np.arange(len(dates)) * 0.8 + 100
    seasonal_effect = np.sin(np.arange(len(dates)) / 30 * 2 * np.pi) * 10 
    close_price = base_price + seasonal_effect + np.random.randn(len(dates)) * 3
    df = pd.DataFrame({'Date': dates, 'Close': close_price})
    print("Menggunakan data dummy 200 hari kerja untuk demonstrasi.")

# --- 2. Pre-processing Data untuk Prophet ---

# 1. Rename kolom sesuai persyaratan Prophet: 'Date' -> 'ds', 'Close' -> 'y'
df_prophet = df.rename(columns={'Date': 'ds', 'Close': 'y'})

# 2. Pastikan kolom 'ds' bertipe datetime
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

print("\nData yang siap untuk Prophet:")
print(df_prophet.tail())

# --- 3. Membagi Data secara Sekuensial (Optional, untuk Testing) ---
# Kita akan gunakan 80% data untuk training dan 20% untuk pengujian/evaluasi
train_size = int(len(df_prophet) * 0.8)
df_train = df_prophet.iloc[:train_size]
df_test = df_prophet.iloc[train_size:]

print(f"\nUkuran Data Latih: {len(df_train)} sampel")
print(f"Ukuran Data Uji: {len(df_test)} sampel")

# --- 4. Inisialisasi dan Pelatihan Model Prophet ---

# Inisialisasi model. Kita asumsikan ada pola musiman mingguan dan tahunan.
# Note: Data harian saham biasanya memiliki pola weekly (lebih tinggi/rendah di hari tertentu)
model = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=False, # Mungkin tidak relevan untuk data pendek
    daily_seasonality=False,
    changepoint_prior_scale=0.05 # Mengatur sensitivitas perubahan tren
)

print("\nMelatih model Prophet...")
model.fit(df_train)
print("Model selesai dilatih.")

# --- 5. Membuat Prediksi (Forecast) ---

# 1. Buat kerangka waktu masa depan (Future DataFrame)
# Kita akan memprediksi selama 30 hari ke depan (termasuk 20% data uji dan hari-hari di masa depan)
future_days = len(df_test) + 30
future = model.make_future_dataframe(periods=future_days, freq='B') # B = Business Days

# 2. Lakukan Prediksi
forecast = model.predict(future)

# --- 6. Visualisasi dan Hasil ---

# 1. Visualisasi Trend dan Seasonality (Komponen Model)
print("\nKomponen Model Prophet (Trend, Weekly, etc.):")
fig_components = model.plot_components(forecast)
plt.show()

# 2. Visualisasi Prediksi Utama
print("\nVisualisasi Prediksi Harga Saham:")
fig_forecast = model.plot(forecast)
# Menambahkan garis pemisah data train dan forecast
plt.axvline(df_test['ds'].min(), color='red', linestyle='--', label='Awal Data Uji')
plt.title(f'Prophet Forecast Harga Saham (Total {future_days} Hari ke Depan)')
plt.xlabel('Tanggal (ds)')
plt.ylabel('Harga Penutupan (y)')
plt.legend()
plt.show()

# 3. Evaluasi (Membandingkan Prediksi dengan Data Uji Aktual)
df_eval = pd.merge(df_test, forecast[['ds', 'yhat']], on='ds', how='left')
df_eval = df_eval.dropna()

if not df_eval.empty:
    mse = mean_squared_error(df_eval['y'], df_eval['yhat'])
    rmse = np.sqrt(mse)
    
    print("\n=== Hasil Evaluasi Prophet pada Data Uji ===")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    # Visualisasi perbandingan y_aktual vs y_pred pada data uji
    plt.figure(figsize=(12, 6))
    plt.plot(df_eval['ds'], df_eval['y'], label='Harga Aktual', color='blue')
    plt.plot(df_eval['ds'], df_eval['yhat'], label='Harga Prediksi (Prophet)', color='red', linestyle='--')
    plt.title('Perbandingan Aktual vs Prediksi Prophet (Data Uji)')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga Penutupan')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("\nTidak ada data uji yang tersedia untuk evaluasi.")