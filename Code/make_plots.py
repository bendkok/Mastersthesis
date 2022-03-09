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


def make_plots(path = Path("fhn_res/fitzhugh_nagumo_res"), model="fitzhugh_nagumo", if_noise=False, params=[0,1]):
    """
    Loads the saved data, and plots them.
    """
    
    sns.set_theme()
    
    filename0 = model+".dat"
    filename1 = model+"_pred.dat"
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
    
    #nosie
    def noise():
        filename5 = model+"_noise.dat"
        data5 = np.loadtxt(os.path.join(path, filename5), delimiter=' ', skiprows=0, dtype=float)
        v_noi = data5[:,1]
        w_noi = data5[:,2]
        
        plt.plot(t, v_noi, label="v")
        plt.plot(t, w_noi, label="w")
        plt.legend(loc="best")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.title("Noisy input data")
        plt.savefig(Path.joinpath( path, "plot_noi0.pdf"))
        plt.show()
        
        plt.plot(t, v_noi, "o", label="Noisy")
        plt.plot(t, v_exe, "r", label="Exact")
        plt.legend(loc="best")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.title("Exact vs. Noisy data v")
        plt.savefig(Path.joinpath( path, "plot_noi1.pdf"))
        plt.show()
    
        plt.plot(t, w_noi, "o", label="Sampled Input")
        plt.plot(t, w_exe, "r", label="Exact")
        plt.legend(loc="best")
        plt.xlabel("Time (ms)")
        plt.ylabel("Current (mA)")
        plt.title("Exact vs. Noisy data w")
        plt.savefig(Path.joinpath( path, "plot_noi2.pdf"))
        plt.show()
    
    #sampled
    def sampled(n_states=2):
        filename6 = model+"_samp.dat"
        data6 = np.loadtxt(os.path.join(path, filename6), delimiter=' ', skiprows=0, dtype=float)
        
        length = len(data6)//(2+n_states)
        idx = np.array(data6[:length]).astype(int)
        t_s = np.array(data6[length:2*length])
        v_s = np.array(data6[2*length:3*length])
        w_s = np.array(data6[3*length:])
        
        plt.plot(t_s, v_s, "o", label="Noisy")
        plt.plot(t, v_exe, "r", label="Exact")
        plt.legend(loc="best")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.title("Exact vs. Sampled data v")
        plt.savefig(Path.joinpath( path, "plot_samp0.pdf"))
        plt.show()
    
        plt.plot(t_s, w_s, "o", label="Sampled Input")
        plt.plot(t, w_exe, "r", label="Exact")
        plt.legend(loc="best")
        plt.xlabel("Time (ms)")
        plt.ylabel("Current (mA)")
        plt.title("Exact vs. Sampled data w")
        plt.savefig(Path.joinpath( path, "plot_samp1.pdf"))
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
    def param_change(params = params, param_names = ["a","b","tau","Iext"], tar=[-.3, 1.1, 20, 0.23]):       
        #plots the change in the prediction of the ode-params.
        for p in params:
            plt.plot(data4[1:,0], data4[1:,p+1], label="Inff. {}".format(param_names[p]))
            plt.plot(data4[1:,0], np.ones(len(data4[1:,0]))*tar[p], "--", label="Tar. {}".format(param_names[p]))
            plt.legend(loc="best")
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.title("Change in {}".format(param_names[p]))
            plt.savefig(Path.joinpath( path, "plot_varchange_{}.pdf".format(param_names[p])))
            plt.show()   
            
        #plots the change in the prediction of the ode-params.
        for p in params:
            plt.plot(data4[1:,0], data4[1:,p+1], label="Inff. {}".format(param_names[p]))
            plt.plot(data4[1:,0], np.ones(len(data4[1:,0]))*tar[p], "--", label="Tar. {}".format(param_names[p]))
            plt.legend(loc="best")
            plt.xlabel("Epoch")
            plt.ylabel("Value")
        plt.title("Change in paramteers")
        plt.yscale("symlog")
        plt.savefig(Path.joinpath( path, "plot_varchange.pdf"))
        plt.show()   
    
    # exact()
    # prediction()
    # nn_prediction()
    # nnb_prediction()
    param_change()   
    try:
        noise()
    except:
        ""
    try:
        sampled()
    except:
        ""
    


if __name__ == "__main__":
    # make_plots(Path("fitzhugh_nagumo_res_feature_onlyb_2"))
    make_plots(Path("fhn_res/fitzhugh_nagumo_res_all_01"), if_noise=True, params=[0,1,2,3])
