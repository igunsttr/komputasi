import pandas as pd
import numpy as np
from sklearn.cluster import KMeans # Model Clustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Persiapan Data Dummy (Fitur Pelanggan) ---
np.random.seed(42)
data = {
    'Annual_Income': np.concatenate([np.random.normal(30, 5, 50), np.random.normal(80, 10, 50), np.random.normal(150, 20, 50)]),
    'Spending_Score': np.concatenate([np.random.normal(80, 10, 50), np.random.normal(40, 10, 50), np.random.normal(20, 10, 50)])
}
df_customer = pd.DataFrame(data)

# --- 2. Standarisasi Data ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_customer)

# --- 3. Menentukan Jumlah Cluster (Metode Elbow) ---
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
# Asumsikan jumlah cluster optimal adalah 3 (berdasarkan data dummy)
optimal_clusters = 3

# --- 4. Pelatihan Model K-Means ---
kmeans_model = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42, n_init=10)
df_customer['Cluster'] = kmeans_model.fit_predict(X_scaled)

print(f"**Clustering (K-Means)**")
print(f"Jumlah optimal cluster yang digunakan: {optimal_clusters}")
print("\nPembagian data per Cluster:")
print(df_customer['Cluster'].value_counts())

# --- 5. Visualisasi Hasil Clustering ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Annual_Income', y='Spending_Score', hue='Cluster', data=df_customer, palette='viridis', style='Cluster')
plt.title('K-Means Clustering: Segmentasi Pelanggan')
plt.xlabel('Pendapatan Tahunan')
plt.ylabel('Skor Pengeluaran')
plt.legend(title='Segmen')
plt.show()