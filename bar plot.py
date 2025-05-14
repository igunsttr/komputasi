import matplotlib.pyplot as plt

# Data
kategori = ['A', 'B', 'C', 'D']
nilai = [3, 8, 1, 10]

# Buat bar plot
plt.bar(kategori, nilai)
plt.xlabel('Kategori')
plt.ylabel('Nilai')
plt.title('Grafik Batang')
plt.show()