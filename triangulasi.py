# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 07:50:35 2024

@author: office
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Baca gambar
img = cv2.imread('C:\\Bahan\Komputasi\\tidak-beraturan.png', cv2.IMREAD_GRAYSCALE)

# Binarisasi gambar (ubah menjadi hitam putih)
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# ... (kode untuk membaca dan membinarisasi gambar sama seperti sebelumnya)

# Temukan kontur (garis tepi) objek
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Ambil kontur terbesar (asumsikan hanya ada satu objek)
cnt = contours[0]

# Ubah kontur menjadi format yang sesuai untuk Delaunay triangulation
points = cnt.reshape(-1, 2)

# Lakukan triangulasi
tri = Delaunay(points)

# Hitung luas setiap segitiga dan jumlahkan
luas = 0
for simplex in tri.simplices:
    p1, p2, p3 = points[simplex]
    # Hitung luas segitiga menggunakan rumus Heron (atau fungsi dari SciPy)
    # ...

print("Luas bentuk tak beraturan (triangulasi):", luas)