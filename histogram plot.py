import matplotlib.pyplot as plt
import numpy as np

# Buat data random
data = np.random.randn(1000)

# Buat histogram
plt.hist(data, bins=30)
plt.xlabel('Nilai')
plt.ylabel('Frekuensi')
plt.title('Histogram')
plt.show()