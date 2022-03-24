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
import pickle


def get_hyperparam_title(path, param_names=["a","b","tau","Iext"]):
    """
    Creates the title for the plots. 
    """
    # hyp = np.loadtxt(os.path.join(path, 'hyperparameters.dat'), delimiter='\n', skiprows=0, dtype=str)
    
    with open(os.path.join(path, "hyperparameters.pkl"), "rb") as a_file:
        hyp0 = pickle.load(a_file)
    # print(hyp0)
    
    # weights = "Weights = [{}, {}, {}]".format(hyp[0].partition(" ")[2], hyp[1].partition(" ")[2], hyp[2].partition(" ")[2])
    weights = "Weights = [{}, {}, {}],  ".format(hyp0['ode_weights'], hyp0['bc_weights'], hyp0['data_weights'])
    #do_t_input_transform 
    
    para = "Fitted parameters = {},  ".format(np.array(param_names)[np.where(hyp0['var_trainable'])])
    
    inp_tran = "\nFeature transformation = t -> ["
    k_vals = hyp0['k_vals'] #hyp[8][9:-1].split(",")
    # print( hyp[-2])
    try:    
        do_t_input_transform = hyp0['do_t_input_transform'] #bool(distutils.util.strtobool( hyp[-2][22:]))
        if do_t_input_transform:
            inp_tran += "t/T, "
    except:
        ""

    inp_tran += "sin(2pi*{}*t)".format(k_vals[0])
    for k in range(1, len(k_vals)):
        inp_tran += ", sin(2pi*{}*t)".format(k_vals[k])
    inp_tran += ']'    
    
    try:    
        noise = "Noise = {}%,".format(hyp0['noise']*100)
        return weights+para+noise+inp_tran
    except:
        return weights+para+inp_tran


def plot_losses(path, 
                do_test_vals=True,
                states = ["v", "w"],
                skiprows=3,
    ):
    """
    Makes one plot of the losses.
    """
    
    sns.set_theme()
    
    inp_dat = np.loadtxt(os.path.join(path, 'loss.dat'), delimiter=' ', skiprows=skiprows, dtype=float)
        
    epochs = inp_dat[:,0]
    
    n_loss = inp_dat.shape[1]//2 +1
    
    train_loss = inp_dat[:,1:n_loss]
    test_loss = inp_dat[:,n_loss:]
        
    parts = ['ODE ', 'BC  ', 'Data']
    # print(states)
    
    #should make 4 plots
    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    axs_falt = axs.flatten()
        
    for i in range(len(parts)):
        axs_falt[0].plot(epochs, np.mean(train_loss[:,i*len(states):i*len(states)+len(states)], axis=1), '-', label='{}'.format(parts[i])) # train_loss[:,i*2] + train_loss[:,i*2+1]
        if do_test_vals:
            if test_loss[:,i*len(states)].all() != 0:    
                axs_falt[0].plot(epochs, np.sum(test_loss[:,i*len(states):i*len(states)+len(states)], axis=1), '--', label='{} Test'.format(parts[i]))
        
        for s in range(len(states)):    
            axs_falt[i+1].plot(epochs, train_loss[:,i*len(states)+s], '-', label='{}'.format(states[s]))
        # axs_falt[i+1].plot(epochs, train_loss[:,i*2+1], '-', label='{}-w'.format(parts[i]))
        
        if do_test_vals:
            if test_loss[:,i*len(states)].all() != 0:   
                for s in range(len(states)):    
                   axs_falt[i+1].plot(epochs, test_loss[:,i*len(states)+s], '--', label='{} Test'.format(states[s]))
                # axs_falt[i+1].plot(epochs, test_loss[:,i*2+1], '--', label='{}-w Test'.format(parts[i]))
        
        mean = np.mean((train_loss[:,i*len(states):i*len(states)+len(states)]), axis=1) 
        axs_falt[i+1].plot(epochs, mean, '--', label='Mean')
        
        if len(states)>2:    
            axs_falt[i+1].legend(ncol=5)
        else:
            axs_falt[i+1].legend()
        axs_falt[i+1].set_yscale('log')
        axs_falt[i+1].set_title("{} loss history".format(parts[i]))
        axs_falt[i+1].set_xlabel("Epoch")
        axs_falt[i+1].set_ylabel("Loss")
        
        
    mean = np.mean(train_loss, axis=1)
    axs_falt[0].plot(epochs, np.zeros_like(epochs)+mean, '--', label='Mean')
    
    axs_falt[0].set_yscale('log')
    axs_falt[0].legend()
    axs_falt[0].set_title("All loss history")
    axs_falt[0].set_xlabel("Epoch")
    axs_falt[0].set_ylabel("Loss")
    
    fig.suptitle(get_hyperparam_title(path), fontsize=15)
    fig.savefig(Path.joinpath( path, "full_loss.pdf"))
    
    plt.show()
    
    # print(train_loss.shape)
    
    for i in range(len(parts)):
        diff = [np.min( np.abs( train_loss[:,i*2]/train_loss[:,i*2+1] ))]
        diff.append( np.mean( np.abs( train_loss[:,i*2]/train_loss[:,i*2+1] )) )
        diff.append( np.max( np.abs( train_loss[:,i*2]/train_loss[:,i*2+1] )) )
        # print("Mean loss difference {}: {}".format(parts[i], diff))
        print("Mean loss {}: {}, {}".format(parts[i], np.mean(train_loss[:,i*2]), np.mean(train_loss[:,i*2+1])))
    print("")
    for i in range(len(parts)):
        print("Min  loss {}: {}, {}".format(parts[i], np.min(train_loss[:,i*2]), np.min(train_loss[:,i*2+1])))
        
    
    diff = np.abs(train_loss[:2] - test_loss[:2])
    print("\nODE train-test diff: ", np.mean(diff), np.min(diff), np.max(diff))
    

def make_one_plot(path, model="fitzhugh_nagumo", states=[1,2], state_names = ["v", "w"]):
    """
    Makes one plot of the prediction.
    """
    
    sns.set_theme()
    
    filename0 = model+".dat"
    filename1 = model+"_pred.dat"
    filename2 = "neural_net_pred_best.dat"
    
    exact = np.loadtxt(os.path.join(path, filename0), delimiter=' ', skiprows=0, dtype=float)
    pred  = np.loadtxt(os.path.join(path, filename1), delimiter=' ', skiprows=0, dtype=float)
    nn    = np.loadtxt(os.path.join(path, filename2), delimiter=' ', skiprows=0, dtype=float)
    
    t, v_exe, w_exe = exact[:,0], exact[:,states[0]], exact[:,states[1]]
    v_pre, w_pre    = pred[:,states[0]], pred[:,states[1]]
    v_nn, w_nn      = nn[:,states[0]], nn[:,states[1]]
    
    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    axs_falt = axs.flatten()
    
    # print(t.shape)
    
    l1, = axs_falt[0].plot(t, v_exe)
    l2, = axs_falt[0].plot(t, v_nn, "r--")
    axs_falt[0].set_title(f"NN's prediction of {state_names[0]} in the best epoch.")
    
    axs_falt[1].plot(t, v_exe)
    axs_falt[1].plot(t, v_pre, "r--")
    axs_falt[1].set_title(f"ODE prediction of {state_names[0]}.")
    
    axs_falt[2].plot(t, w_exe)
    axs_falt[2].plot(t, w_nn, "r--")
    axs_falt[2].set_title(f"NN's prediction of {state_names[1]} in the best epoch.")
    
    axs_falt[3].plot(t, w_exe)
    axs_falt[3].plot(t, w_pre, "r--")
    axs_falt[3].set_title(f"ODE prediction of {state_names[1]}.")
    
    for i in range(4):
        axs_falt[i].set_xlabel("Time (ms)")
        axs_falt[i].set_ylabel("Potential (mV)")
        # axs_falt[i].grid()
        
    axs_falt[2].set_ylabel("Current (mA)")
    axs_falt[3].set_ylabel("Current (mA)")
    
    fig.suptitle(get_hyperparam_title(path), fontsize=15)
    fig.legend((l1,l2), ("Exact", "Prediction"), bbox_to_anchor=(0.5,0.5), loc="center", ncol=1)
    
    fig.savefig(Path.joinpath( path, "full_plot.pdf"))
    
    plt.show()
    

def make_samp_plot(path, model="fitzhugh_nagumo", states=[1,2], state_names = ["v", "w"]):
    """
    Makes one plot of the prediction vs. sampled points.
    """
    
    sns.set_theme()
    
    # filename0 = model+".dat"
    filename1 = model+"_pred.dat"
    filename2 = "neural_net_pred_best.dat"
    filename3 = model+"_samp.dat"
    
    # exact = np.loadtxt(os.path.join(path, filename0), delimiter=' ', skiprows=0, dtype=float)
    pred  = np.loadtxt(os.path.join(path, filename1), delimiter=' ', skiprows=0, dtype=float)
    nn    = np.loadtxt(os.path.join(path, filename2), delimiter=' ', skiprows=0, dtype=float)
    samp  = np.loadtxt(os.path.join(path, filename3), delimiter=' ', skiprows=0, dtype=float)
    
    s_samp = len(samp)//(2+len(states))
    
    # t_s, w_exe, v_exe = exact[s_samp:2*s_samp], exact[2*s_samp:3*s_samp], exact[3*s_samp:]
    # t, v_exe, w_exe = exact[:,0], exact[:,states[0]], exact[:,states[1]]
    t, v_pre = pred[:,0], pred[:,states[0]]
    v_nn      = nn[:,states[0]]
    
    if len(states)>1:
        w_pre = pred[:,states[1]]
        w_nn  = nn[:,states[1]]
        t_s, v_s, w_s   = samp[s_samp:2*s_samp], samp[2*s_samp::2], samp[2*s_samp+1::2]
    else:
        t_s, v_s        = samp[s_samp:2*s_samp], samp[2*s_samp:]
    
    
    if len(states)>1:
        fig, axs = plt.subplots(2, 2, figsize=(14,10))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(17,7))
    axs_falt = axs.flatten()
    
    # print(t.shape)
    
    l1, = axs_falt[0].plot(t_s, v_s, "o")
    l2, = axs_falt[0].plot(t, v_nn, "r")
    axs_falt[0].set_title(f"NN's prediction of {state_names[0]} in the best epoch.")
    
    axs_falt[1].plot(t_s, v_s, "o")
    axs_falt[1].plot(t, v_pre, "r")
    axs_falt[1].set_title(f"ODE prediction of {state_names[0]}.")
    
    if len(states)>1:
        axs_falt[2].plot(t_s, w_s, "o")
        axs_falt[2].plot(t, w_nn, "r")
        axs_falt[2].set_title(f"NN's prediction of {state_names[1]} in the best epoch.")
        
        axs_falt[3].plot(t_s, w_s, "o")
        axs_falt[3].plot(t, w_pre, "r")
        axs_falt[3].set_title(f"ODE prediction of {state_names[1]}.")
    
    for i in range(2*len(states)):
        axs_falt[i].set_xlabel("Time (ms)")
        axs_falt[i].set_ylabel("Potential (mV)")
        # axs_falt[i].grid()
    
    if len(states)>1:
        axs_falt[2].set_ylabel("Current (mA)")
        axs_falt[3].set_ylabel("Current (mA)")
    
    tit = get_hyperparam_title(path)
    
    fig.suptitle(tit, fontsize=15)
    fig.legend((l1,l2), ("Sampled", "Prediction"), bbox_to_anchor=(0.5,0.5), loc="center", ncol=1)
    
    fig.savefig(Path.joinpath( path, "full_samp.pdf"))
    
    plt.show()
    

def make_state_plot(path, model="beeler_reuter", use_nn = False):
    """
    Makes one plot of all the predictions.
    """
    
    sns.set_theme()
    
    states = "m, h, j, Cai, d, f, x1, V".split(", ")
    
    
    filename0 = model+".dat"
    if use_nn:    
        filename1 = "neural_net_pred_best.dat"
    else:
        filename1 = model+"_pred.dat"
    
    exact = np.loadtxt(os.path.join(path, filename0), delimiter=' ', skiprows=0, dtype=float)
    pred  = np.loadtxt(os.path.join(path, filename1), delimiter=' ', skiprows=0, dtype=float)
    
    
    fig, axs = plt.subplots(4, 2, figsize=(18,25))
    axs_falt = axs.flatten()
    
    for s in range(0,len(states)):
        l2, = axs_falt[s].plot(pred[:,0], pred[:,s+1], "r--")
        l1, = axs_falt[s].plot(exact[:,0], exact[:,s+1])
        if use_nn:
            axs_falt[s].set_title(f"NN prediction of {states[s]}.")
        else:    
            axs_falt[s].set_title(f"ODE prediction of {states[s]}.")
        axs_falt[s].set_xlabel("Time (ms)")
        axs_falt[s].set_ylabel("Voltage (mV)") #change
    
    fig.suptitle(get_hyperparam_title(path), fontsize=15)
    fig.legend((l1,l2), ("Exact", "Prediction"), bbox_to_anchor=(0.5,0.5), loc="center", ncol=1)
    
    if use_nn:    
        fig.savefig(Path.joinpath( path, "full_nn.pdf"))
    else:
        fig.savefig(Path.joinpath( path, "full_states.pdf"))
    
    plt.show()


def main():
    
    # path = Path("fhn_res/fhn_res_clus/fhn_res_s-01_v-a_n0_e20")
    # make_one_plot(path)
    # plot_losses(path)
    # plot_losses(path, do_test_vals=False)
    # make_samp_plot(path, states=[1,2])
    for dir in os.listdir('./fhn_res/fhn_res_clus'):
        path = Path("fhn_res/fhn_res_clus/{}".format(dir))
        with open(os.path.join(path, "hyperparameters.pkl"), "rb") as a_file:
            data = pickle.load(a_file)
        states = data['observed_states']
        states = [s+1 for s in states]
        
        # make_one_plot(path, states=states)
        make_one_plot(path)
        plot_losses(path, do_test_vals=False)
        make_samp_plot(path, states=states)
        # make_samp_plot(path)
    
    
    
    # path = Path("br_res/br_res_09")
    
    # sns.set_theme()
    # make_one_plot(path, "beeler_reuter", states=[8,4], state_names = ["V", "Cai"])
    # plot_losses(path, states = "m, h, j, Cai, d, f, x1, V".split(", "), skiprows=1)
    
    # make_state_plot(path)
    # make_state_plot(path, use_nn=True)
    
    
    
if __name__ == "__main__":
    main()