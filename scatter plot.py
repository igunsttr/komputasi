import matplotlib.pyplot as plt
import numpy as np

# Buat data random
np.random.seed(19680801)
x = np.random.rand(100)
y = np.random.rand(100)

# Buat scatter plot
plt.scatter(x, y)
plt.xlabel('Variabel X')
plt.ylabel('Variabel Y')
plt.title('Grafik Sebar')
plt.show()