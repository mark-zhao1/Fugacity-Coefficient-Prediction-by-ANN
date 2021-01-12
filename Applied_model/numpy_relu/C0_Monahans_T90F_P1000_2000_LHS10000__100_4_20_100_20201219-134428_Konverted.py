"""
  Generated using Konverter: https://github.com/ShaneSmiskol/Konverter
"""

import numpy as np

wb = np.load('C:/Users/markz/PycharmProjects/Ln_phi_model/Applied_model/numpy_relu/C0_Monahans_T90F_P1000_2000_LHS10000__100_4_20_100_20201219-134428_Konverted_weights.npz', allow_pickle=True)
w, b = wb['wb']

def predict(x):
  x = np.array(x, dtype=np.float32)
  l0 = np.dot(x, w[0]) + b[0]
  l0 = np.where(l0 > 0, l0, l0 * 0.1)
  l1 = np.dot(l0, w[1]) + b[1]
  l1 = np.where(l1 > 0, l1, l1 * 0.1)
  l2 = np.dot(l1, w[2]) + b[2]
  l2 = np.where(l2 > 0, l2, l2 * 0.1)
  l3 = np.dot(l2, w[3]) + b[3]
  l3 = np.where(l3 > 0, l3, l3 * 0.1)
  l4 = np.dot(l3, w[4]) + b[4]
  return l4
