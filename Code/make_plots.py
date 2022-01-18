# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 00:48:56 2021

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import re
import seaborn as sns


def make_plots(path = Path("fhn_res/fitzhugh_nagumo_res")):
    """
    Loads the saved data, and plots them.
    """
    
    sns.set_theme()
    
    filename0 = "fitzhugh_nagumo.dat"
    filename1 = "fitzhugh_nagumo_pred.dat"
    filename2 = "neural_net_pred_last.dat"
    filename3 = "neural_net_pred_best.dat"
    filename4 = "variables.dat"
    
    data0 = np.loadtxt(os.path.join(path, filename0), delimiter=' ', skiprows=0, dtype=float)
    data1 = np.loadtxt(os.path.join(path, filename1), delimiter=' ', skiprows=0, dtype=float)
    data2 = np.loadtxt(os.path.join(path, filename2), delimiter=' ', skiprows=0, dtype=float)
    data3 = np.loadtxt(os.path.join(path, filename3), delimiter=' ', skiprows=0, dtype=float)
    
    with open(os.path.join(path, filename4)) as f:
        lines = f.read().splitlines()
    
    li = []
    for i in range(len(lines)):
        l = lines[i].split(" ")
        l = [re.sub("\[|\]|,", "", a ) for a in l]
        li.append(l)
        
    data4 = np.asarray(li, dtype=np.float64, order='C')
    
    t = data0[:,0]
    v_exe = data0[:,1]
    w_exe = data0[:,2]
    
    v_pre = data1[:,1]
    w_pre = data1[:,2]
    
    v_nn = data2[:,1]
    w_nn = data2[:,2]
    
    v_nnb = data3[:,1]
    w_nnb = data3[:,2]
    
        
    #exact
    def exact():
        #plots the exact values
        plt.plot(t, v_exe, label="v")
        plt.plot(t, w_exe, label="w")
        plt.legend(loc="best")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.title("Input data")
        plt.savefig(Path.joinpath( path, "plot_exe.pdf"))
        plt.show()
        
        
    #prediction
    def prediction():
        #plots the result with the prediction for the ODE-params.
        plt.plot(t, v_pre, label="v")
        plt.plot(t, w_pre, label="w")
        plt.legend(loc="best")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.title("Prediction")
        plt.savefig(Path.joinpath( path, "plot_pred.pdf"))
        plt.show()

        plt.plot(t, v_exe, label="Exact")
        plt.plot(t, v_pre, "r--", label="Learned")
        plt.legend(loc="best")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.title("Exact vs. Predicted v")
        plt.savefig(Path.joinpath( path, "plot_comp0.pdf"))
        plt.show()
    
        plt.plot(t, w_exe, label="Exact")
        plt.plot(t, w_pre, "r--", label="Learned")
        plt.legend(loc="best")
        plt.xlabel("Time (ms)")
        plt.ylabel("Current (mA)")
        plt.title("Exact vs. Predicted w")
        plt.savefig(Path.joinpath( path, "plot_comp1.pdf"))
        plt.show()
        
        plt.plot(v_pre, w_pre, label="v against w")
        # plt.legend(loc="best")
        plt.xlabel("v (mV)")
        plt.ylabel("w (mA)")
        plt.title("Phase space of the FitzHugh–Nagumo model")
        plt.savefig(Path.joinpath( path, "plot_pha.pdf"))
        plt.show()
    
    
    #Neural Network
    #prediction
    def nn_prediction():
        #plots the nn's prediction in the last epoch
        plt.plot(t, v_nn, label="v")
        plt.plot(t, w_nn, label="w")
        plt.legend(loc="best")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.title("Neural Network Prediction Last Epoch")
        plt.savefig(Path.joinpath( path, "plot_pred_nn.pdf"))
        plt.show()
        
        plt.plot(t, v_exe, label="Exact")
        plt.plot(t, v_nn, "r--", label="NN Prediction Last Epoch")
        plt.legend(loc="best")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.title("Exact vs. NN Predicted v")
        plt.savefig(Path.joinpath( path, "plot_comp_nn0.pdf"))
        plt.show()
        
        plt.plot(t, w_exe, label="Exact")
        plt.plot(t, w_nn, "r--", label="NN Prediction Last Epoch")
        plt.legend(loc="best")
        plt.xlabel("Time (ms)")
        plt.ylabel("Current (mA)")
        plt.title("Exact vs. NN Predicted w")
        plt.savefig(Path.joinpath( path, "plot_comp_nn1.pdf"))
        plt.show()
    

    # #prediction
    def nnb_prediction():
        #plots the nn's prediction in the best epoch
        plt.plot(t, v_nnb, label="v")
        plt.plot(t, w_nnb, label="w")
        plt.legend(loc="best")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.title("Neural Network Prediction Best Epoch")
        plt.savefig(Path.joinpath( path, "plot_pred_nnb.pdf"))
        # plt.savefig(path + "/plot_pred.pdf")
        plt.show()
        
        plt.plot(t, v_exe, label="Exact")
        plt.plot(t, v_nnb, "r--", label="NN Prediction Best Epoch")
        plt.legend(loc="best")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.title("Exact vs. NN Predicted v")
        plt.savefig(Path.joinpath( path, "plot_comp_nnb0.pdf"))
        # plt.savefig(path + "/plot_comp0.pdf")
        plt.show()
        
        plt.plot(t, w_exe, label="Exact")
        plt.plot(t, w_nnb, "r--", label="NN Prediction Best Epoch")
        plt.legend(loc="best")
        plt.xlabel("Time (ms)")
        plt.ylabel("Current (mA)")
        plt.title("Exact vs. NN Predicted w")
        plt.savefig(Path.joinpath( path, "plot_comp_nnb1.pdf"))
        # plt.savefig(path + "/plot_comp1.pdf")
        plt.show()
    
    
    #change in parameters
    def param_change(params = [1], param_names = ["a", "b", "tau", "Iext"]):        
        #plots the change in the prediction of the ode-params.
        for p in params:
            plt.plot(data4[:,0], data4[:,p+1], label="Aproximation")
            plt.plot(data4[:,0], np.ones(len(data4[:,0]))*1.1, "--", label="Target")
            plt.legend(loc="best")
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.title("Change in {}".format(param_names[p]))
            plt.savefig(Path.joinpath( path, "plot_varchange_{}.pdf".format(param_names[p])))
            plt.show()    
    
    # exact()
    # prediction()
    # nn_prediction()
    # nnb_prediction()
    param_change()   


if __name__ == "__main__":
    # make_plots(Path("fitzhugh_nagumo_res_feature_onlyb_2"))
    make_plots(Path("fhn_res/fitzhugh_nagumo_res_bas10_2_50"))
