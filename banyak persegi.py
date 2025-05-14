# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 07:41:12 2024

@author: office
"""

import numpy as np

# Buat array dengan panjang dan lebar beberapa persegi panjang
panjang = np.array([3, 4, 5])
lebar = np.array([2, 6, 1])

# Hitung luas semua persegi panjang sekaligus
luas = panjang * lebar

print("Luas masing-masing persegi panjang:", luas)