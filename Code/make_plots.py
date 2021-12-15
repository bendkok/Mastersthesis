# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 00:48:56 2021

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import deepxde as dde
from deepxde.backend import tf
import re
import seaborn as sns


def make_plots(path = Path("fhn_res/fitzhugh_nagumo_res")):

    sns.set_theme()
    
    # filename0 = "fitzhugh_nagumo.dat"
    # filename1 = "fitzhugh_nagumo_pred.dat"
    # filename2 = "neural_net_pred_last.dat"
    # filename3 = "neural_net_pred_best.dat"
    filename4 = "variables.dat"
    
    # data0 = np.loadtxt(os.path.join(path, filename0), delimiter=' ', skiprows=0, dtype=float)
    # data1 = np.loadtxt(os.path.join(path, filename1), delimiter=' ', skiprows=0, dtype=float)
    # data2 = np.loadtxt(os.path.join(path, filename2), delimiter=' ', skiprows=0, dtype=float)
    # data3 = np.loadtxt(os.path.join(path, filename3), delimiter=' ', skiprows=0, dtype=float)
    
    with open(os.path.join(path, filename4)) as f:
        lines = f.read().splitlines()
    
    li = []
    for i in range(len(lines)):
        l = lines[i].split(" ")
        l = [re.sub("\[|\]|,", "", a ) for a in l]
        li.append(l)
        
    data4 = np.asarray(li, dtype=np.float64, order='C')
    
    # t = data0[:,0]
    # v_exe = data0[:,1]
    # w_exe = data0[:,2]
    
    # t_ = data1[:,0]
    # v_pre = data1[:,1]
    # w_pre = data1[:,2]
    
    # t__ = data2[:,0]
    # v_nn = data2[:,1]
    # w_nn = data2[:,2]
    
    # t___ = data3[:,0]
    # v_nnb = data3[:,1]
    # w_nnb = data3[:,2]
    
    # print(v_pre.shape, w_pre.shape, v_exe.shape, w_exe.shape)
    
    # if np.all(t != t_):
    #     print("t1 not equal")
    # if np.all(t != t__):
    #     print("t2 not equal")
        
    #exact
    # plt.plot(t, v_exe, label="v")
    # plt.plot(t, w_exe, label="w")
    # plt.legend(loc="best")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Voltage (mV)")
    # plt.title("Input data")
    # plt.savefig(Path.joinpath( path, "plot_exe.pdf"))
    # plt.show()
        
    # #prediction
    # plt.plot(t, v_pre, label="v")
    # plt.plot(t, w_pre, label="w")
    # plt.legend(loc="best")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Voltage (mV)")
    # plt.title("Prediction")
    # plt.savefig(Path.joinpath( path, "plot_pred.pdf"))
    # # plt.savefig(path + "/plot_pred.pdf")
    # plt.show()
    
    # plt.plot(t, v_exe, label="Exact")
    # plt.plot(t, v_pre, "r--", label="Learned")
    # plt.legend(loc="best")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Voltage (mV)")
    # plt.title("Exact vs. Predicted v")
    # plt.savefig(Path.joinpath( path, "plot_comp0.pdf"))
    # # plt.savefig(path + "/plot_comp0.pdf")
    # plt.show()
    
    # plt.plot(t, w_exe, label="Exact")
    # plt.plot(t, w_pre, "r--", label="Learned")
    # plt.legend(loc="best")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Current (mA)")
    # plt.title("Exact vs. Predicted w")
    # plt.savefig(Path.joinpath( path, "plot_comp1.pdf"))
    # # plt.savefig(path + "/plot_comp1.pdf")
    # plt.show()
    
    # plt.plot(v_pre, w_pre, label="v against w")
    # # plt.legend(loc="best")
    # plt.xlabel("v (mV)")
    # plt.ylabel("w (mA)")
    # plt.title("Phase space of the FitzHugh–Nagumo model")
    # plt.savefig(Path.joinpath( path, "plot_pha.pdf"))
    # # plt.savefig(path + "/plot_pha.pdf")
    # plt.show()
    
    
    #Neural Network
    #prediction
    # plt.plot(t, v_nn, label="v")
    # plt.plot(t, w_nn, label="w")
    # plt.legend(loc="best")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Voltage (mV)")
    # plt.title("Neural Network Prediction Last Epoch")
    # plt.savefig(Path.joinpath( path, "plot_pred_nn.pdf"))
    # # plt.savefig(path + "/plot_pred.pdf")
    # plt.show()
    
    # plt.plot(t, v_exe, label="Exact")
    # plt.plot(t, v_nn, "r--", label="NN Prediction Last Epoch")
    # plt.legend(loc="best")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Voltage (mV)")
    # plt.title("Exact vs. NN Predicted v")
    # plt.savefig(Path.joinpath( path, "plot_comp_nn0.pdf"))
    # # plt.savefig(path + "/plot_comp0.pdf")
    # plt.show()
    
    # plt.plot(t, w_exe, label="Exact")
    # plt.plot(t, w_nn, "r--", label="NN Prediction Last Epoch")
    # plt.legend(loc="best")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Current (mA)")
    # plt.title("Exact vs. NN Predicted w")
    # plt.savefig(Path.joinpath( path, "plot_comp_nn1.pdf"))
    # # plt.savefig(path + "/plot_comp1.pdf")
    # plt.show()
    

    
    # #prediction
    # plt.plot(t, v_nnb, label="v")
    # plt.plot(t, w_nnb, label="w")
    # plt.legend(loc="best")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Voltage (mV)")
    # plt.title("Neural Network Prediction Best Epoch")
    # plt.savefig(Path.joinpath( path, "plot_pred_nnb.pdf"))
    # # plt.savefig(path + "/plot_pred.pdf")
    # plt.show()
    
    # plt.plot(t, v_exe, label="Exact")
    # plt.plot(t, v_nnb, "r--", label="NN Prediction Best Epoch")
    # plt.legend(loc="best")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Voltage (mV)")
    # plt.title("Exact vs. NN Predicted v")
    # plt.savefig(Path.joinpath( path, "plot_comp_nnb0.pdf"))
    # # plt.savefig(path + "/plot_comp0.pdf")
    # plt.show()
    
    # plt.plot(t, w_exe, label="Exact")
    # plt.plot(t, w_nnb, "r--", label="NN Prediction Best Epoch")
    # plt.legend(loc="best")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Current (mA)")
    # plt.title("Exact vs. NN Predicted w")
    # plt.savefig(Path.joinpath( path, "plot_comp_nnb1.pdf"))
    # # plt.savefig(path + "/plot_comp1.pdf")
    # plt.show()
    
    
    #change in parameters
    # plt.plot(data4[:,0], data4[:,1], label="Aproximation")
    # plt.plot(data4[:,0], np.ones(len(data4[:,0]))*(-.3), "--", label="Target")
    # plt.legend(loc="best")
    # plt.xlabel("Epoch")
    # plt.ylabel("Value")
    # plt.title("Change in a")
    # plt.savefig(Path.joinpath( path, "plot_varchange_a.pdf"))
    # plt.show()
    
    plt.plot(data4[:,0], data4[:,2], label="Aproximation")
    plt.plot(data4[:,0], np.ones(len(data4[:,0]))*1.1, "--", label="Target")
    plt.legend(loc="best")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Change in b")
    plt.savefig(Path.joinpath( path, "plot_varchange_b.pdf"))
    plt.show()
    
    # plt.plot(data4[:,0], data4[:,3], label="Aproximation")
    # plt.plot(data4[:,0], np.ones(len(data4[:,0]))*20, "--", label="Target")
    # plt.legend(loc="best")
    # plt.xlabel("Epoch")
    # plt.ylabel("Value")
    # plt.title("Change in τ")
    # plt.savefig(Path.joinpath( path, "plot_varchange_t.pdf"))
    # plt.show()
    
    # plt.plot(data4[:,0], data4[:,4], label="Aproximation")
    # plt.plot(data4[:,0], np.ones(len(data4[:,0]))*.23, "--", label="Target")
    # plt.legend(loc="best")
    # plt.xlabel("Epoch")
    # plt.ylabel("Value")
    # plt.title("Change in Iext")
    # plt.savefig(Path.joinpath( path, "plot_varchange_I.pdf"))
    # plt.show()
    
    
    
    # data_t = np.linspace(0, 999, 1000)
    # model = create_dummy_model([.013/2])
    # model.compile("adam", lr=1e-3)
    # model.restore(get_model_restore_path(path))
    # nn_pred = model.predict(data_t.reshape(-1,1)) 
    # plt.plot(data_t, nn_pred) 
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Voltage (mV)")
    # plt.title("Neural Network Prediction")
    # plt.savefig(Path.joinpath( path, "plot_pred_nn_k0.pdf"))
    # plt.show()
    
    
    
def create_dummy_model(k_vals):
    
    net = dde.maps.FNN(
        # layer_size=[1, 128, 128, 128, 2],
        layer_size=[1] + [128]*3 + [2],
        activation="swish",
        kernel_initializer="Glorot normal",
    )
    
    def feature_transform(t):
        features = [] # np.zeros(len(k_vals) + 1)
        # features.append(t) #[0] = t
        for k in range(len(k_vals)):
            features.append( tf.sin(k_vals[k] * 2*np.pi*t) )
        return tf.concat(features, axis=1)

    net.apply_feature_transform(feature_transform)

    def output_transform(t, y):
        # Weights in the output layer are chosen as the magnitudes
        # of the mean values of the ODE solution
        # return data_y[0] + tf.math.tanh(t) * tf.constant([0.1, 0.1]) * y
        return [0,0] + tf.math.tanh(t) * tf.constant([0.1, 0.1]) * y

    net.apply_output_transform(output_transform)
    
    var_list=[-.3, 1.1, 20, 0.23]
    def ODE(t, y):
        v1 = y[:, 0:1] - y[:, 0:1] ** 3 / 3 - y[:, 1:2] + var_list[3]
        v2 = (y[:, 0:1] - var_list[0] - var_list[1] * y[:, 1:2]) / var_list[2]
        return [
            tf.gradients(y[:, 0:1], t)[0] - v1,
            tf.gradients(y[:, 1:2], t)[0] - v2,
        ]
    
    data = dde.data.PDE(  
        dde.geometry.TimeDomain(0, 999),
        ODE, 
        []
        )
    
    model = dde.Model(data, net)
    return model
    


def get_model_restore_path(savename):
    #reads form the checkpoint text file
    with open(os.path.join(savename,"model/checkpoint"), 'r') as reader:
        inp = reader.read()
        restore_from = inp.split(" ")[1].split('"')[1]
    return os.path.join(savename,"model", restore_from)
   



if __name__ == "__main__":
    # make_plots(Path("fitzhugh_nagumo_res_feature_onlyb_2"))
    make_plots(Path("fhn_res/fitzhugh_nagumo_res_bas10_2_80"))
