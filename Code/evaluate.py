# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:25:10 2022

@author: benda
"""

import imp
from typing_extensions import runtime
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
import pickle


def evaluate(path, model="fitzhugh_nagumo", true_param = [-0.3, 1.1, 20, 0.23], runtime=6273.53030538559):
    """
    Evaluates the results. 
    """
    
    #imports and organizes the data
    filename0 = model+".dat"
    filename1 = model+"_pred.dat"
    filename2 = "neural_net_pred_best.dat"
    filename4 = "variables.dat"
    
    with open(os.path.join(path, "hyperparameters.pkl"), "rb") as a_file:
        hyp0 = pickle.load(a_file)
    fitted_para = hyp0['var_trainable']


    exact = np.loadtxt(os.path.join(path, filename0), delimiter=' ', skiprows=0, dtype=float)
    pred  = np.loadtxt(os.path.join(path, filename1), delimiter=' ', skiprows=0, dtype=float)
    nn    = np.loadtxt(os.path.join(path, filename2), delimiter=' ', skiprows=0, dtype=float)
    
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

    
    #finds the MRE for the fitted parameters
    # ind = np.where(found_param!=true_param)[0]
    ind = np.where(fitted_para)[0]
    param_re  = np.abs(found_param-true_param)/np.abs(true_param)
    param_mre = np.mean(param_re[ind])
    param_mae = mean_absolute_error(np.array(true_param)[ind], np.array(found_param)[ind]) #legacy
    print("\nActual parameters:   {}. \nInffered parameters: {}. \nParameter rel. err.: {}.\n\nParameter MRE: {}.\n".format(true_param, found_param.tolist(), param_re, param_mre))
    # print(param_re)
    
    #regular mse
    ode_mse = mean_squared_error(exact, pred)
    nn_mse = mean_squared_error(exact, nn)
    ode_nn_mse = mean_squared_error(pred, nn)
    
    print("ODE v. real MSE: {}".format(ode_mse))
    print("NN  v. real MSE: {}".format(nn_mse))
    print("ODE v. NN   MSE: {}".format(ode_nn_mse))
    
    
    results = dict(
        true_param=np.array(true_param)[ind], found_param=np.array(found_param)[ind], param_mae=param_mae,
        param_re=param_re, param_mre=param_mre,
        ode_mse=ode_mse, nn_mse=nn_mse, ode_nn_mse=ode_nn_mse,
    )
    if runtime != None:
        results.update(runtime=runtime)
    
    a_file = open(os.path.join(path, "evaluation.pkl"), "wb") 
    pickle.dump(results, a_file)    
    a_file.close()
    
    with open(os.path.join(path, "evaluation.dat"),'w') as data: 
        for key, value in results.items(): 
            data.write('%s: %s\n' % (key, value))
            
    return param_mre
    # if param_mre > .01:
    #     return True
    # else:
    #     return False
            
    

if __name__ == "__main__":
    
    # print(os.listdir('./fhn_res/fhn_res_clus'))

    for dir in os.listdir('./fhn_res/fhn_res_clus'):
        path = Path("fhn_res/fhn_res_clus/{}".format(dir))
        try:
            with open(os.path.join(path, "evaluation.pkl"), "rb") as a_file:
                data = pickle.load(a_file)
            runtime = data['runtime']
        except:
            runtime = None
        # print(runtime)
        evaluate(path = path, runtime=runtime)
    
    
    
    
