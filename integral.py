
from scipy.integrate import quad

def fungsi(x):
    return x**2

# Mendefinisikan batas integral
a = 0
b = 2

# Menghitung integral
result, error = quad(fungsi, a, b)

print("Hasil integral:", result)