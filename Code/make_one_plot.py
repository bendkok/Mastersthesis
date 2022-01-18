# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:56:23 2021

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import seaborn as sns
import distutils


def get_hyperparam_title(path):
    hyp = np.loadtxt(os.path.join(path, 'hyperparameters.dat'), delimiter='\n', skiprows=0, dtype=str)
    
    
    weights = "Weights = [{}, {}, {}]".format(hyp[0].partition(" ")[2], hyp[1].partition(" ")[2], hyp[2].partition(" ")[2])
    #do_t_input_transform 
    
    inp_tran = "Feature transformation = t -> ["
    k_vals = hyp[7][9:-1].split(",")
    try:    
        do_t_input_transform = bool(distutils.util.strtobool( hyp[10][22:]))
        if do_t_input_transform:
            inp_tran += "t, "
    except:
        ""

    inp_tran += "sin(2pi*{}*t)".format(k_vals[0])
    for k in range(1, len(k_vals)):
        inp_tran += ", sin(2pi*{}*t)".format(k_vals[k])
    inp_tran += ']'    
    
    return weights+", "+inp_tran


def plot_losses(path, do_test_vals=True):
    
    sns.set_theme()
    
    inp_dat = np.loadtxt(os.path.join(path, 'loss.dat'), delimiter=' ', skiprows=3, dtype=float)
    
    
    epochs = inp_dat[:,0]
    
    n_loss = inp_dat.shape[1]//2 +1
    
    train_loss = inp_dat[:,1:n_loss]
    test_loss = inp_dat[:,n_loss:]
    
    parts = ['ODE', 'BC', 'Data']
    
    #should make 4 plots
    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    axs_falt = axs.flatten()
    
    for i in range(len(parts)):
        axs_falt[0].plot(epochs, train_loss[:,i*2] + train_loss[:,i*2+1], '-', label='{}'.format(parts[i]))
        if do_test_vals:
            if test_loss[:,i*2].all() != 0:    
                axs_falt[0].plot(epochs, test_loss[:,i*2] + test_loss[:,i*2+1], '--', label='{} Test'.format(parts[i]))
        
        
        axs_falt[i+1].plot(epochs, train_loss[:,i*2], '-', label='{}-v'.format(parts[i]))
        axs_falt[i+1].plot(epochs, train_loss[:,i*2+1], '-', label='{}-w'.format(parts[i]))
        
        if do_test_vals:
            if test_loss[:,i*2].all() != 0:            
                axs_falt[i+1].plot(epochs, test_loss[:,i*2], '--', label='{}-v Test'.format(parts[i]))
                axs_falt[i+1].plot(epochs, test_loss[:,i*2+1], '--', label='{}-w Test'.format(parts[i]))
        
        mean = np.mean((train_loss[:,i*2:i*2+2]), axis=1) 
        axs_falt[i+1].plot(epochs, mean, '--', label='Mean')
        
        
        axs_falt[i+1].legend()
        axs_falt[i+1].set_yscale('log')
        axs_falt[i+1].set_title("{} loss history".format(parts[i]))
        axs_falt[i+1].set_xlabel("Epoch")
        axs_falt[i+1].set_ylabel("Loss")
        
        
    mean = np.mean(np.mean((train_loss, test_loss), axis=0), axis=1)
    axs_falt[0].plot(epochs, np.zeros_like(epochs)+mean, '--', label='Mean')
    
    axs_falt[0].set_yscale('log')
    axs_falt[0].legend()
    axs_falt[0].set_title("All loss history")
    axs_falt[0].set_xlabel("Epoch")
    axs_falt[0].set_ylabel("Loss")
    
    fig.suptitle(get_hyperparam_title(path), fontsize=15)
    fig.savefig(Path.joinpath( path, "full_loss.pdf"))
    
    plt.show()
    
    
    for i in range(len(parts)):
        diff = [np.min( np.abs( train_loss[:,i*2]/train_loss[:,i*2+1] ))]
        diff.append( np.mean( np.abs( train_loss[:,i*2]/train_loss[:,i*2+1] )) )
        diff.append( np.max( np.abs( train_loss[:,i*2]/train_loss[:,i*2+1] )) )
        print("Mean loss difference {}: {}".format(parts[i], diff))
        print("Mean loss {}: {}, {}".format(parts[i], np.mean(train_loss[:,i*2]), np.mean(train_loss[:,i*2+1])))
    
    
    diff = np.abs(train_loss[:2] - test_loss[:2])
    print("ODE train-test diff: ", np.mean(diff), np.min(diff), np.max(diff))
    

def make_one_plot(path):
    
    sns.set_theme()
    
    filename0 = "fitzhugh_nagumo.dat"
    filename1 = "fitzhugh_nagumo_pred.dat"
    filename2 = "neural_net_pred_best.dat"
    
    exact = np.loadtxt(os.path.join(path, filename0), delimiter=' ', skiprows=0, dtype=float)
    pred  = np.loadtxt(os.path.join(path, filename1), delimiter=' ', skiprows=0, dtype=float)
    nn    = np.loadtxt(os.path.join(path, filename2), delimiter=' ', skiprows=0, dtype=float)
    
    t, v_exe, w_exe = exact[:,0], exact[:,1], exact[:,2]
    v_pre, w_pre    = pred[:,1], pred[:,2]
    v_nn, w_nn      = nn[:,1], nn[:,2]
    
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
        axs_falt[i].set_xlabel("Time (ms)")
        axs_falt[i].set_ylabel("Voltage (mV)")
        axs_falt[i].grid()
        
    axs_falt[2].set_ylabel("Current (mA)")
    axs_falt[3].set_ylabel("Current (mA)")
    
    
    fig.suptitle(get_hyperparam_title(path), fontsize=15)
    fig.legend((l1,l2), ("Exact", "Prediction"), bbox_to_anchor=(0.5,0.5), loc="center", ncol=1)
    
    
    fig.savefig(Path.joinpath( path, "full_plot.pdf"))
    
    plt.show()
    



def main():
    
    # path = Path("glycolysis_res")
    path = Path("fhn_res/fitzhugh_nagumo_res_bas10_2_170")
    
    sns.set_theme()
    make_one_plot(path)
    plot_losses(path)
    
    
    
if __name__ == "__main__":
    main()