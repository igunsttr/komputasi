import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # Model Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- 1. Persiapan Data Deret Waktu Dummy (Contoh Harga Saham) ---
np.random.seed(42)
data_len = 100
# Membuat deret waktu tren + noise
series = np.arange(data_len) * 0.5 + np.sin(np.arange(data_len) / 10) * 10 + np.random.normal(0, 2, data_len)
df_ts = pd.DataFrame(series, columns=['Value'])

# --- 2. Teknik Feature Engineering (Lagging) ---
# Mengubah masalah Time Series menjadi masalah Regresi
time_step = 10 # Menggunakan 10 langkah waktu sebelumnya sebagai fitur (lag)

def create_lagged_features(df, lag_steps):
    """Membuat fitur lagging dari deret waktu."""
    df_lag = df.copy()
    for i in range(1, lag_steps + 1):
        df_lag[f'Lag_{i}'] = df_lag['Value'].shift(i)
    # Hapus baris dengan nilai NaN (baris awal)
    df_lag.dropna(inplace=True)
    
    X_lag = df_lag.drop('Value', axis=1) # Fitur: Nilai lag
    y_lag = df_lag['Value']             # Target: Nilai saat ini (yang akan diprediksi)
    return X_lag, y_lag

X, y = create_lagged_features(df_ts, time_step)

# --- 3. Pembagian dan Skala Data ---
# Pembagian data (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. Pelatihan Model Regresi Sklearn (Random Forest) ---
# Random Forest Regressor sering memberikan hasil yang baik untuk masalah regresi
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

print("\nMemulai Pelatihan Model Random Forest...")
rf_model.fit(X_train_scaled, y_train)
print("Pelatihan Selesai.")

# --- 5. Prediksi dan Evaluasi ---
y_pred = rf_model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n✅ Root Mean Squared Error (RMSE): **{rmse:.4f}**")

# --- 6. Visualisasi Hasil Forecasting ---
# Gabungkan y_test dan y_pred untuk plot
actual_series = np.concatenate([y_train, y_test])
predicted_series = np.concatenate([y_train, y_pred]) # Menggunakan y_train di awal karena tidak diprediksi

plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(actual_series)), actual_series, label='Nilai Sebenarnya', color='blue')
# Plot prediksi hanya pada segmen test
test_start_index = len(y_train)
plt.plot(np.arange(test_start_index, len(actual_series)), y_pred, label='Prediksi Random Forest', color='red')
plt.axvline(x=test_start_index, color='gray', linestyle='--', label='Batas Train/Test')

plt.title('Forecasting Deret Waktu dengan Sklearn Random Forest (Lagging)')
plt.xlabel('Langkah Waktu')
plt.ylabel('Nilai')
plt.legend()
plt.show()