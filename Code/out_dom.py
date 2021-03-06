# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 00:23:57 2022

@author: benda
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from fhn_pinn import fitzhugh_nagumo_model
import seaborn as sns

sns.set_theme()


def out_dom(saveloc="out_dom"):
    
    path = Path("out_dom/expe_10")
    
    filename4 = "variables.dat"
    filename0 = "neural_net_pred_out_dom.dat"
    filename1 = "neural_net_pred_best.dat"
    
    nn_pred = np.loadtxt(os.path.join(path, filename0), delimiter=' ', skiprows=0, dtype=float)
    nn_org = np.loadtxt(os.path.join(path, filename1), delimiter=' ', skiprows=0, dtype=float)
    
    
    with open(os.path.join(path, filename4)) as f:
        lines = f.read().splitlines()
        
    li = []
    for i in range(len(lines)):
        l = lines[i].split(" ")
        l = [re.sub("\[|\]|,", "", a ) for a in l]
        li.append(l)
        
    #the found params
    try:
        data4 = np.asarray(li, dtype=np.float64, order='C')
    except:
        data4 = np.asarray(li[:-1], dtype=np.float64, order='C')
    found_param = data4[-1,1:]
    
    # print(found_param)
    
    t0 = np.linspace(000, 999, 2000)
    t1 = np.linspace(1000, 1999, 1000)
    t = np.append(t0,t1)
    
    # for i in range(len(t1)):    
    #     nn_pred[i,1:] = nn_org[0,1:] + np.tanh(t1[i]) * np.array([1., .1]) * nn_pred[i,1:]
    
    t_full = np.concatenate((t0[1300:], t1))
    
    ode_pred = fitzhugh_nagumo_model(t, *found_param)
    ode_pred0 = ode_pred[:2000] #fitzhugh_nagumo_model(t0, *found_param)
    ode_pred1 = ode_pred[2000:] #fitzhugh_nagumo_model(t1, *found_param)
    real = fitzhugh_nagumo_model(t)
    
    # print(ode_pred1.shape, ode_pred0[1300:].shape)
    nn_full = np.concatenate((nn_org[1300:], nn_pred), axis=0)
    ode_full = np.concatenate((ode_pred0[1300:], ode_pred1), axis=0)
    
    # print(nn_pred.shape, ode_pred.shape)
    
    
    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    axs_falt = axs.flatten()
    
    # l1, = axs_falt[0].plot(t0[1300:], nn_org[1300:,1])
    # l2, = axs_falt[0].plot(t1, nn_pred[:,1])#, "r--")
    # l3, = axs_falt[0].plot(t[1300:], real[1300:,0], '--', zorder=0)
    # axs_falt[0].set_title("NN's prediction of v.")
    
    l1, = axs_falt[0].plot(t_full, nn_full[:,1], linewidth=2.)
    l2, = axs_falt[0].plot(t_full, real[1300:,0], 'r--', zorder=5, linewidth=2.0)
    axs_falt[0].set_title("NN's prediction of v.")
    axs_falt[0].axvline(x=1000, color='black', zorder=15, linestyle="-", label='axvline - full height')
    
    # axs_falt[1].plot(t0[1300:], nn_org[1300:,2])
    # axs_falt[1].plot(t1, nn_pred[:,2])#, "r--")
    # axs_falt[1].plot(t[1300:], real[1300:,1], '--', zorder=0)
    # axs_falt[1].set_title("NN's prediction of w.")
    
    axs_falt[1].plot(t_full, nn_full[:,2], linewidth=2.)#, "r--")
    axs_falt[1].plot(t_full, real[1300:,1], 'r--', zorder=10, linewidth=2.0)
    axs_falt[1].set_title("NN's prediction of w.")
    axs_falt[1].axvline(x=1000, color='black', zorder=15, linestyle="-", label='axvline - full height')
    
    # axs_falt[2].plot(t0[1300:], ode_pred0[1300:,0])
    # axs_falt[2].plot(t1, ode_pred1[:,0])#, "r--")
    # axs_falt[2].plot(t[1300:], real[1300:,0], '--', zorder=0)
    
    axs_falt[2].plot(t_full, ode_full[:,0], linewidth=2.)#, "r--")
    axs_falt[2].plot(t_full, real[1300:,0], 'r--', zorder=10, linewidth=2.)
    axs_falt[2].set_title("ODE prediction of v.")
    axs_falt[2].axvline(x=1000, color='black', zorder=15, linestyle="-", label='axvline - full height')
    
    # axs_falt[3].plot(t0[1300:], ode_pred0[1300:,1])
    # axs_falt[3].plot(t1, ode_pred1[:,1])#, "r--")
    # axs_falt[3].plot(t[1300:], real[1300:,1], '--', zorder=0)
    
    axs_falt[3].plot(t_full, ode_full[:,1], linewidth=2.)#, "r--")
    axs_falt[3].plot(t_full, real[1300:,1], 'r--', zorder=10, linewidth=2.0)
    axs_falt[3].set_title("ODE prediction of w.")
    axs_falt[3].axvline(x=1000, color='black', zorder=15, linestyle="-", label='axvline - full height')
    
    for i in range(4):
        axs_falt[i].set_xlabel("Time (ms)")
        axs_falt[i].set_ylabel("Potential (mV)")
        # axs_falt[i].grid()
        
    axs_falt[1].set_ylabel("Current (mA)")
    axs_falt[3].set_ylabel("Current (mA)")
    
    fig.legend((l1,l2), ("Prediction", "Real"), bbox_to_anchor=(0.5,0.5), loc="center", ncol=1)
    
    # fig.savefig(saveloc+"/out_dom_pred.pdf")
    
    plt.show()
    
    


if __name__ == "__main__":
    out_dom()

