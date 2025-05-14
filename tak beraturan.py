# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:20:48 2025

@author: office
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 07:43:03 2024

@author: office
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar
img = cv2.imread('C:\\Bahan\Komputasi\\tidak-beraturan.png', cv2.IMREAD_GRAYSCALE)

# Binarisasi gambar (ubah menjadi hitam putih)
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Hitung jumlah piksel putih (objek)
jumlah_piksel_objek = cv2.countNonZero(thresh)

# Asumsikan ukuran piksel adalah 1x1 satuan luas
luas = jumlah_piksel_objek

# Tampilkan gambar asli dan hasil binarisasi
plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Gambar Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(thresh, cmap='gray')
plt.title('Hasil Binarisasi'), plt.xticks([]), plt.yticks([])
plt.show()

print("Luas bentuk tak beraturan (perkiraan):", luas)