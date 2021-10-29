# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:56:23 2021

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def make_one_plot(path):
    
    filename0 = "fitzhugh_nagumo.dat"
    filename1 = "fitzhugh_nagumo_pred.dat"
    filename2 = "neural_net_pred_best.dat"
    
    
    exact = np.loadtxt(os.path.join(path, filename0), delimiter=' ', skiprows=0, dtype=float)
    pred = np.loadtxt(os.path.join(path, filename1), delimiter=' ', skiprows=0, dtype=float)
    nn = np.loadtxt(os.path.join(path, filename2), delimiter=' ', skiprows=0, dtype=float)
    
    t, v_exe, w_exe = exact[:,0], exact[:,1], exact[:,2]
    t1, v_pre, w_pre = pred[:,0], pred[:,1], pred[:,2]
    t2, v_nn, w_nn = nn[:,0], nn[:,1], nn[:,2]
    
    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    axs_falt = axs.flatten()
    
    l1, = axs_falt[0].plot(t, v_exe)
    l2, = axs_falt[0].plot(t, v_nn, "r--")
    axs_falt[0].set_title("NN's prediction of v in the best epoch.")
    
    axs_falt[1].plot(t, v_exe)
    axs_falt[1].plot(t, v_pre, "r--")
    axs_falt[1].set_title("ODE prediction of v.")
    
    axs_falt[2].plot(t, w_exe)
    axs_falt[2].plot(t, w_nn, "r--")
    axs_falt[2].set_title("title")
    axs_falt[2].set_title("NN's prediction of w in the best epoch.")
    
    axs_falt[3].plot(t, w_exe)
    axs_falt[3].plot(t, w_pre, "r--")
    axs_falt[3].set_title("ODE prediction of w.")
    
    for i in range(4):
        axs_falt[i].set_xlabel("Time (s? or ms?)")
        axs_falt[i].set_ylabel("Voltage (mV)")
        # axs_falt[i].grid()
        
    axs_falt[2].set_ylabel("Current (mA)")
    axs_falt[3].set_ylabel("Current (mA)")
    
    fig.legend((l1,l2), ("Exact", "Prediction"), bbox_to_anchor=(0.5,0.5), loc="center", ncol=1)
    
    fig.savefig(Path.joinpath( path, "full_plot.pdf"))
    plt.show()
    



def main():
    
    make_one_plot(Path("fhn_res/fitzhugh_nagumo_res_feature_onlyb_6"))
    
    
    
if __name__ == "__main__":
    main()