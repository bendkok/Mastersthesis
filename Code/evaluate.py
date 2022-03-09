# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:25:10 2022

@author: benda
"""

import numpy as np
import scipy as sp
from scipy import fft, signal
import matplotlib.pyplot as plt
import os
from pathlib import Path
import seaborn as sns
import distutils
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from julia import base


def evaluate(path, model="fitzhugh_nagumo", states=[1,2], true_param = [-0.3, 1.1, 20, 0.23], runtime=6273.53030538559):
    """
    Evaluates the results. 
    """
    
    #imports and organizes the data
    filename0 = model+".dat"
    filename1 = model+"_pred.dat"
    filename2 = "neural_net_pred_best.dat"
    filename4 = "variables.dat"
    
    exact = np.loadtxt(os.path.join(path, filename0), delimiter=' ', skiprows=0, dtype=float)
    pred  = np.loadtxt(os.path.join(path, filename1), delimiter=' ', skiprows=0, dtype=float)
    nn    = np.loadtxt(os.path.join(path, filename2), delimiter=' ', skiprows=0, dtype=float)
    
    # t, v_exe, w_exe = exact[:,0], exact[:,states[0]], exact[:,states[1]]
    # v_pre, w_pre    = pred[:,states[0]], pred[:,states[1]]
    # v_nn, w_nn      = nn[:,states[0]], nn[:,states[1]]
    
    with open(os.path.join(path, filename4)) as f:
        lines = f.read().splitlines()
    
    li = []
    for i in range(len(lines)):
        l = lines[i].split(" ")
        l = [re.sub("\[|\]|,", "", a ) for a in l]
        li.append(l)
        
    #the found params
    data4 = np.asarray(li, dtype=np.float64, order='C')
    found_param = data4[-1,1:]

    
    #finds the MSE for the fitted parameters
    ind = np.where(found_param!=true_param)[0]
    param_mae = mean_absolute_error(np.array(true_param)[ind], np.array(found_param)[ind])
    print("Actual parameters:   {}. \nInffered parameters: {}. \nParameter MAE: {}.\n".format(true_param, found_param.tolist(), param_mae))
    
    #regular mse
    ode_mse = mean_squared_error(exact, pred)
    nn_mse = mean_squared_error(exact, nn)
    ode_nn_mse = mean_squared_error(pred, nn)
    
    print("ODE v. real MSE: {}".format(ode_mse))
    print("NN v. real MSE: {}".format(nn_mse))
    print("ODE v. NN values MSE: {}".format(ode_nn_mse))
    
    
    results = dict(
        true_param=np.array(true_param)[ind], found_param=np.array(found_param)[ind], param_mae=param_mae,
        ode_mse=ode_mse, nn_mse=nn_mse, ode_nn_mse=ode_nn_mse,
    )
    if runtime != None:
        results.update(runtime=runtime)
    import pickle #try this later
    a_file = open(os.path.join(path, "evaluation.pkl"), "wb") 
    pickle.dump(results, a_file)    
    a_file.close()
    
    with open(os.path.join(path, "evaluation.dat"),'w') as data: 
        for key, value in results.items(): 
            data.write('%s: %s\n' % (key, value))
    
    
    # ff_exe = fft.rfft(exact[:,2])
    # ff_ode = fft.rfft(pred[:,2])
    # freq = fft.rfftfreq(t.shape[-1])
    
    
    # sig = signal.find_peaks(np.abs(ff_exe).real, prominence=None)[0]
    # sig0 = signal.find_peaks(np.abs(ff_ode).real, prominence=None)[0]
    
    # print(mean_squared_error(np.real(ff_exe), np.real(ff_ode)))
    
    # # print(sig, ff_exe.real[sig])
    
    
    # plt.plot(freq, np.real(ff_exe))
    # plt.plot(freq, np.real(ff_ode))
    # plt.plot([0.0173, 0.0173], [ff_exe.real.min(), np.real(ff_ode).max()], "--")
    # plt.plot(freq[sig], np.real(ff_exe)[sig], "o")
    # plt.plot(freq[sig0], np.real(ff_ode)[sig0], "o")
    # plt.yscale("log")
    # # plt.axis([-.005,.1, ff_exe.real.min()-10, np.abs(ff_ode).max()+10])
    # # plt.grid()
    # plt.plot()
    # plt.show()
    
        
    
    

if __name__ == "__main__":
    
    evaluate(path = Path("fhn_res/fitzhugh_nagumo_res_ab_00"))
    
    
    
    
