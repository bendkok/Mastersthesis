# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 17:30:28 2021

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt

filename0 = ".\\glycolysis.dat"
filename1 = "./glycolysis_pred.dat"

data0 = np.loadtxt(filename0, delimiter=' ', skiprows=0, dtype=float)
data1 = np.loadtxt(filename1, delimiter=' ', skiprows=0, dtype=float)

plt.plot(data0[:,0], data0[:,1])
plt.plot(data1[:,0], data1[:,1])
plt.show()

plt.plot(data0[:,0], data0[:,2])
plt.plot(data1[:,0], data1[:,2])
plt.show()

plt.plot(data0[:,0], data0[:,3])
plt.plot(data1[:,0], data1[:,3])
plt.show()

plt.plot(data0[:,0], data0[:,4])
plt.plot(data1[:,0], data1[:,4])
plt.show()

plt.plot(data0[:,0], data0[:,5])
plt.plot(data1[:,0], data1[:,5])
plt.show()

plt.plot(data0[:,0], data0[:,6])
plt.plot(data1[:,0], data1[:,6])
plt.show()

plt.plot(data0[:,0], data0[:,7])
plt.plot(data1[:,0], data1[:,7])
plt.show()
