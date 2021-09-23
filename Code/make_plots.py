# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 00:48:56 2021

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt
import os


path = "fitzhugh_nagumo_res"

filename0 = "fitzhugh_nagumo.dat"
filename1 = "fitzhugh_nagumo_pred.dat"

data0 = np.loadtxt(os.path.join(path, filename0), delimiter=' ', skiprows=0, dtype=float)
data1 = np.loadtxt(os.path.join(path, filename1), delimiter=' ', skiprows=0, dtype=float)

t = data0[:,0]
v_exe = data0[:,1]
w_exe = data0[:,2]

t_ = data0[:,0]
v_pre = data1[:,1]
w_pre = data1[:,2]

print(v_pre.shape, w_pre.shape, v_exe.shape, w_exe.shape)

if np.all(t != t_):
    print("ts not equal")
    
#exact
plt.plot(t, v_exe, label="v")
plt.plot(t, w_exe, label="w")
plt.legend(loc="best")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Input data")
plt.savefig(path + "/plot_exe.pdf")
plt.show()
    
#prediction
plt.plot(t, v_pre, label="v")
plt.plot(t, w_pre, label="w")
plt.legend(loc="best")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Prediction")
plt.savefig(path + "/plot_pred.pdf")
plt.show()

plt.plot(t, v_exe, label="Exact")
plt.plot(t, v_pre, "r--", label="Learned")
plt.legend(loc="best")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Exact vs. Predicted v")
plt.savefig(path + "/plot_comp0.pdf")
plt.show()

plt.plot(t, w_exe, label="Exact")
plt.plot(t, w_pre, "r--", label="Learned")
plt.legend(loc="best")
plt.xlabel("Time (ms)")
plt.ylabel("Current (mA)")
plt.title("Exact vs. Predicted w")
plt.savefig(path + "/plot_comp1.pdf")
plt.show()

plt.plot(v_pre, w_pre, label="v against w")
# plt.legend(loc="best")
plt.xlabel("v (mV)")
plt.ylabel("w (mA)")
plt.title("Phase space of the FitzHugh–Nagumo model")
plt.savefig(path + "/plot_pha.pdf")
plt.show()

