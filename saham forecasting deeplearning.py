import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# --- 1. Konfigurasi dan Pembuatan Fungsi ---

# Panjang urutan waktu masa lalu yang digunakan untuk memprediksi nilai berikutnya
SEQUENCE_LENGTH = 60 
# Rasio pembagian data (80% Training, 20% Testing)
TEST_RATIO = 0.2

def create_sequences(data, sequence_length):
    """
    Mengubah array data menjadi set urutan (X) dan nilai target (y)
    untuk model machine learning tradisional.
    """
    X, y = [], []
    for i in range(sequence_length, len(data)):
        # X: urutan data dari [i - sequence_length] hingga [i-1]
        X.append(data[i-sequence_length:i, 0])
        # y: data pada posisi [i] (nilai yang akan diprediksi)
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# --- 2. Memuat Data dan Pre-processing ---

print("--- 1. Memuat dan Mempersiapkan Data ---")
file_path = 'nama_file_saham.csv'

# --- MEMBUAT DATA DUMMY JIKA FILE TIDAK DITEMUKAN ---
try:
    df = pd.read_csv(file_path)
    if 'Date' in df.columns and 'Close' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').set_index('Date')
        data = df['Close'].values.reshape(-1, 1)
    else:
        raise ValueError("File harus memiliki kolom 'Date' dan 'Close'")
except (FileNotFoundError, ValueError):
    print("Gagal memuat file. Menggunakan data dummy 200 hari kerja.")
    dates = pd.date_range(start='2020-01-01', periods=200, freq='B')
    base_price = np.arange(len(dates)) * 0.5 + 50
    close_price = base_price + np.random.randn(len(dates)) * 5
    df = pd.DataFrame({'Close': close_price}, index=dates)
    data = close_price.reshape(-1, 1)

print(f"Total data poin: {len(data)}")

# Normalisasi Data (Scaling 0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Pembagian Data Sequential
train_size = int(len(data_scaled) * (1 - TEST_RATIO))
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# Memastikan data test memiliki setidaknya cukup sampel untuk satu urutan
if len(test_data) <= SEQUENCE_LENGTH:
    print(f"\n[PERINGATAN] Data Uji terlalu kecil ({len(test_data)} sampel) untuk SEQUENCE_LENGTH={SEQUENCE_LENGTH}. Mengubah SEQUENCE_LENGTH menjadi 10.")
    SEQUENCE_LENGTH = 10
    
# Membuat Sequence (Fitur Lagging)
X_train, y_train = create_sequences(train_data, SEQUENCE_LENGTH)
X_test, y_test = create_sequences(test_data, SEQUENCE_LENGTH)

print(f"Bentuk Data Latih X_train: {X_train.shape}")
print(f"Bentuk Data Uji X_test: {X_test.shape}")


# --- 3. Membangun dan Melatih Model SVR ---

if X_test.shape[0] == 0:
    print("\n[ERROR FATAL] Data Uji kosong setelah pembuatan sequence. Tidak dapat melanjutkan prediksi.")
else:
    print("\n--- 2. Pelatihan Model Support Vector Regressor (SVR) ---")
    
    # Inisialisasi Model SVR
    # Menggunakan kernel RBF untuk pemodelan non-linear
    model_svr = SVR(kernel='rbf', C=100, gamma=0.1) 
    
    # Melatih Model
    model_svr.fit(X_train, y_train)
    print("Model SVR selesai dilatih.")

    
    # --- 4. Prediksi dan Evaluasi ---

    print("\n--- 3. Prediksi dan Evaluasi ---")
    
    # Melakukan Prediksi
    predictions_scaled = model_svr.predict(X_test)
    
    # Inverse Transform (Mengembalikan ke Skala Harga Asli)
    # Penting: Gunakan .reshape(-1, 1) karena scaler mengharapkan input 2D
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluasi Kinerja
    rmse_svr = np.sqrt(mean_squared_error(y_test_original, predictions))

    print(f"Root Mean Squared Error (RMSE) pada Skala Asli: {rmse_svr:.2f}")

    
    # --- 5. Visualisasi Hasil ---

    print("\n--- 4. Visualisasi Hasil ---")
    
    # Membuat index tanggal yang benar untuk data uji
    # Kita harus menyesuaikan index awal karena SEQUENCE_LENGTH hari terbuang di awal test_data
    start_index_test = train_size + SEQUENCE_LENGTH
    df_results = pd.DataFrame({
        'Actual': y_test_original.flatten(),
        'Predicted': predictions.flatten()
    }, index=df.index[start_index_test:])

    plt.figure(figsize=(14, 7))
    plt.plot(df_results.index, df_results['Actual'], label='Harga Aktual', color='blue')
    plt.plot(df_results.index, df_results['Predicted'], label='Harga Prediksi (SVR)', color='red', linestyle='--')
    

    plt.title(f'Prediksi Harga Saham Menggunakan SVR (RMSE: {rmse_svr:.2f})')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga Penutupan')
    plt.legend()
    plt.grid(True)
    plt.show()