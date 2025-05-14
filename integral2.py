
from sympy import symbols, integrate

x = symbols('x')
#f = x**2
#f = x**2-1
f=1/5*x**3

# Menghitung integral secara simbolik
result = integrate(f, (x, 0, 8))

print("Hasil integral:", result)

