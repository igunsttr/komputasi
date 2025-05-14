# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 07:40:54 2024

@author: office
"""

import math

def hitung_luas_lingkaran(jari_jari):
  luas = math.pi * jari_jari**2
  return luas

# Input dari pengguna
jari_jari = float(input("Masukkan jari-jari: "))

# Hitung luas dan tampilkan hasil
hasil = hitung_luas_lingkaran(jari_jari)
print("Luas lingkaran adalah:", hasil)