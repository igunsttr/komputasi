# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 07:39:40 2024

@author: office
"""

import math

def hitung_luas_persegi_panjang(panjang, lebar):
  luas = panjang * lebar
  return luas

# Input dari pengguna
panjang = float(input("Masukkan panjang: "))
lebar = float(input("Masukkan lebar: "))

# Hitung luas dan tampilkan hasil
hasil = hitung_luas_persegi_panjang(panjang, lebar)
print("Luas persegi panjang adalah:", hasil)
