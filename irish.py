import pandas as pd
import matplotlib.pyplot as plt

# Baca data dari file CSV (sesuaikan dengan nama file dan path Anda)
df = pd.read_csv('trending.csv')

# Asumsikan kolom tanggal bernama 'tanggal' dan kolom nilai pencarian bernama 'nilai'
# Sesuaikan dengan nama kolom yang ada di file CSV Anda
plt.plot(df['time'], df['qty'])
plt.xlabel('Tanggal')
plt.ylabel('Nilai Pencarian')
plt.title('Tren Pencarian')
plt.grid(False)
plt.show()