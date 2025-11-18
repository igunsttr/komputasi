import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR # Model Regresi
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- 1. Persiapan Data Dummy (Mirip Dataset Harga Rumah) ---
np.random.seed(42)
X = np.random.rand(100, 3) * 100 # Fitur: Ukuran, Jumlah Kamar, Usia
y = 10 + X[:, 0] * 2 + X[:, 1] * 5 - X[:, 2] * 0.5 + np.random.normal(0, 10, 100) # Target: Harga
df = pd.DataFrame(X, columns=['Size', 'Rooms', 'Age'])
df['Price'] = y

# --- 2. Pembagian dan Skala Data ---
X_train, X_test, y_train, y_test = train_test_split(df[['Size', 'Rooms', 'Age']], df['Price'], test_size=0.2, random_state=42)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

# --- 3. Pelatihan Model Regresi (SVR) ---
svr_model = SVR(kernel='rbf', C=100)
svr_model.fit(X_train_scaled, y_train_scaled)

# --- 4. Prediksi dan Evaluasi ---
y_pred_scaled = svr_model.predict(X_test_scaled)
# Kembalikan prediksi ke skala asli
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Metrik Regresi
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"**Regresi (SVR)**")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# Plot Hasil
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Regresi SVR: Nilai Sebenarnya vs. Prediksi')
plt.xlabel('Nilai Sebenarnya')
plt.ylabel('Prediksi')
plt.show()