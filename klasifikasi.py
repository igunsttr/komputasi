import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score

# --- 1. Pemuatan Data (Asumsi: Diabetes Dataset dari Kaggle) ---
try:
    df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
except FileNotFoundError:
    print("File 'diabetes.csv' tidak ditemukan. Membuat dummy data.")
    # Dummy data jika file tidak ditemukan
    data = np.random.rand(100, 9)
    df = pd.DataFrame(data, columns=[f'Feature_{i}' for i in range(8)] + ['Outcome'])
    df['Outcome'] = np.random.randint(0, 2, 100)

X = df.drop('Outcome', axis=1) # Target/Label
y = df['Outcome'] # Fitur

# --- 2. Pembagian dan Standarisasi Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Pelatihan Model (MLPClassifier) ---
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    activation='relu',
    solver='adam',
    random_state=42
)

print("Memulai Pelatihan Model...")
mlp.fit(X_train_scaled, y_train)

# --- 4. Prediksi dan Evaluasi ---
y_pred = mlp.predict(X_test_scaled)
y_proba = mlp.predict_proba(X_test_scaled)[:, 1] # Probabilitas untuk kelas positif (1)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Akurasi Model: **{accuracy:.4f}**")
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))

# --- 5. Visualisasi Hasil Klasifikasi ---

# A. Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Diabetes (0)', 'Diabetes (1)'],
            yticklabels=['Non-Diabetes (0)', 'Diabetes (1)'])
plt.title('Confusion Matrix')
plt.ylabel('Nilai Sebenarnya')
plt.xlabel('Prediksi Model')
plt.show() 

# B. Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Garis Acak')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show() 
