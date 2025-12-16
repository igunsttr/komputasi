# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 12:56:36 2025

@author: sttrc
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Memuat Dataset
try:
    df = pd.read_csv('nama_file_saham.csv')
except FileNotFoundError:
    print("Gagal memuat file. Pastikan 'nama_file_saham.csv' ada di direktori yang benar.")
    # Membuat data dummy jika file tidak ditemukan
    data = {
        'Open': np.random.rand(100) * 100,
        'High': np.random.rand(100) * 100 + 5,
        'Low': np.random.rand(100) * 100 - 5,
        'Close': np.random.rand(100) * 100,
        'Volume': np.random.randint(10000, 50000, 100)
    }
    df = pd.DataFrame(data)
    print("Menggunakan data dummy untuk demonstrasi.")

# Melihat beberapa baris pertama
print("\nData Saham Awal:")
print(df.head())

# 2. Pemilihan Fitur (X) dan Target (y)
# Fitur (X): Kita gunakan 'Open', 'High', 'Low', dan 'Volume'
X = df[['Open', 'High', 'Low', 'Volume']]

# Target (y): Kita gunakan harga 'Close' (Harga Penutupan)
y = df['Close']

# 3. Membagi Data menjadi Data Latih (Training) dan Data Uji (Testing)
# Rasio 80% untuk training, 20% untuk testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nUkuran Data Latih (X_train): {X_train.shape}")
print(f"Ukuran Data Uji (X_test): {X_test.shape}")

# 4. Inisialisasi Model Regresi Linear
model = LinearRegression()


# 5. Melatih Model (Fitting)
print("\nMelatih model Regresi Linear...")
model.fit(X_train, y_train)

# 6. Melakukan Prediksi pada Data Uji
y_pred = model.predict(X_test)
print("Model selesai dilatih.")
# 7. Evaluasi Kinerja Model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=== Hasil Evaluasi Model Regresi Linear ===")
print(f"Koefisien Model (Bobot Fitur): {model.coef_}")
print(f"Intercept (Konstanta): {model.intercept_}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}") # Semakin mendekati 1, semakin baik model menjelaskan variasi data.

# 8. Visualisasi (Hanya untuk beberapa data uji pertama)
plt.figure(figsize=(12, 6))

# Karena data saham adalah time-series, kita bisa memplotnya berdasarkan indeks waktu
# Namun, karena train_test_split mengacak data, kita gunakan plot sebar untuk y_test vs y_pred
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)

# Garis ideal (prediksi = aktual)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Garis Prediksi Ideal (y=x)')

plt.title('Perbandingan Harga Aktual vs. Harga Prediksi (Regresi Linear)')
plt.xlabel('Harga Aktual (y_test)')
plt.ylabel('Harga Prediksi (y_pred)')
plt.grid(True)
plt.legend()
plt.show()