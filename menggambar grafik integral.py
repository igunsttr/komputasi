import numpy 
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Definisi fungsi
def f(x):
    return numpy.sin(x)

# Batas integral
a = 0
b = numpy.pi

# Menghitung integral
hasil, eror = quad(f, a, b)

# Membuat plot
x = numpy.linspace(a, b, 50)
print("X:",x)
y = f(x)
plt.plot(x, y)
plt.fill_between(x, y, where=(x >= a) & (x <= b), alpha=0.5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Grafik Fungsi dan Daerah Integral')
plt.text(numpy.pi/2, 0.5, f'Luas: {hasil:.2f}')
plt.show()
print("hasil:",hasil)
print("error:",eror)
