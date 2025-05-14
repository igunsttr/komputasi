import matplotlib.pyplot as plt
import numpy as np

# Buat data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Buat plot
plt.plot(x, y)
plt.xlabel('Sumbu X')
plt.ylabel('Sumbu Y')
plt.title('Grafik Sinus')
plt.grid(True)
plt.show()