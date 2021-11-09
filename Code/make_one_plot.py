# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:56:23 2021

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import seaborn as sea


def get_hyperparam_title(path):
    hyp = np.loadtxt(os.path.join(path, 'hyperparameters.dat'), delimiter='\n', skiprows=0, dtype=str)
    # print(hyp)
    
    weights = "Weights = [{}, {}, {}]".format(hyp[0][12:], hyp[1][14:], hyp[2][13:])
    # print(weights)
    
    inp_tran = "Feature transformation = t -> ["
    k_vals = hyp[7][9:-1].split(",")
    # k_vals = "0.01, 0.02".split(", ")
    inp_tran += "sin(2pi*{}*t)".format(k_vals[0])
    for k in range(1, len(k_vals)):
        inp_tran += ", sin(2pi*{}*t)".format(k_vals[k])
    inp_tran += ']'
    # print(inp_tran)
    # print(weights+", "+inp_tran)
    
    
    return weights+", "+inp_tran

def plot_losses(path):
    
    inp_dat = np.loadtxt(os.path.join(path, 'loss.dat'), delimiter=' ', skiprows=3, dtype=float)
    
    
    epochs = inp_dat[:,0]
    
    train_loss = inp_dat[:,1:7]
    test_loss = inp_dat[:,7:]
    print(test_loss.shape, train_loss.shape)
    
    parts = ['BC', 'Data', 'ODE']
    
    # [[bc], [data], [ode]]
    #should make 4 plots
    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    axs_falt = axs.flatten()
    
    for i in range(int(train_loss.shape[1]/2)):
        axs_falt[0].plot(epochs, train_loss[:,i*2] + train_loss[:,i*2+1], '-', label='{} Train'.format(parts[i]))
        axs_falt[0].plot(epochs, test_loss[:,i*2] + test_loss[:,i*2+1], '--', label='{} Test'.format(parts[i]))
        
        axs_falt[i+1].plot(epochs, train_loss[:,i*2] + train_loss[:,i*2+1], '-', label='{} Train'.format(parts[i]))
        axs_falt[i+1].plot(epochs, test_loss[:,i*2] + test_loss[:,i*2+1], '--', label='{} Test'.format(parts[i]))
        axs_falt[i+1].legend()
        axs_falt[i+1].set_title("{} loss history".format(parts[i]))
        axs_falt[i+1].set_xlabel("Epoch")
        axs_falt[i+1].set_ylabel("Loss")
        
    axs_falt[0].set_yscale('log')
    axs_falt[0].legend()
    axs_falt[0].set_title("All loss history")
    axs_falt[0].set_xlabel("Epoch")
    axs_falt[0].set_ylabel("Loss")
    plt.show()
    
    

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
    axs_falt[2].set_title("NN's prediction of w in the best epoch.")
    
    axs_falt[3].plot(t, w_exe)
    axs_falt[3].plot(t, w_pre, "r--")
    axs_falt[3].set_title("ODE prediction of w.")
    
    for i in range(4):
        axs_falt[i].set_xlabel("Time (s)")
        axs_falt[i].set_ylabel("Voltage (mV)")
        # axs_falt[i].grid()
        
    axs_falt[2].set_ylabel("Current (mA)")
    axs_falt[3].set_ylabel("Current (mA)")
    
    
    fig.suptitle(get_hyperparam_title(path), fontsize=15)
    fig.legend((l1,l2), ("Exact", "Prediction"), bbox_to_anchor=(0.5,0.5), loc="center", ncol=1)
    
    
    
    fig.savefig(Path.joinpath( path, "full_plot.pdf"))
    
    plt.show()
    



def main():
    
    path = Path("fhn_res/fitzhugh_nagumo_res_feature_onlyb_7")
    
    # make_one_plot(path)
    
    plot_losses(path)
    
    
    
if __name__ == "__main__":
    main()