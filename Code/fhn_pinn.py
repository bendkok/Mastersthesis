# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:13:48 2021

@author: benda
"""

import numpy as np
from scipy.integrate import odeint

import deepxde as dde
from deepxde.backend import tf

from glycolysis import pinn
from fitzhugh_nagumo import fitzhugh_nagumo

import os
# os.system("python glycosis.py")
# os.system("python fitzhugh_nagumo.py")

def fitzhugh_nagumo_model(
        t,
        a = -0.3,
        b = 1.4,
        tau = 20,
        Iext = 0.23
    ):
    
    x0 = [0,0]
    
    return odeint(fitzhugh_nagumo, x0, t, args=(a, b, tau, Iext))


def main():
    #t = np.arange(0, 10, 0.005)[:, None]
    noise = 0.
    
    savename = "fitzhugh_nagumo_res"
    
    # Data
    
    t = np.linspace(0, 999, 1000)[:, None]
    
    y = fitzhugh_nagumo_model(np.ravel(t))
    np.savetxt(os.path.join(savename, "fitzhugh_nagumo.dat"), np.hstack((t, y)))
    # Add noise
    if noise > 0:
        std = noise * y.std(0)
        y[1:-1, :] += np.random.normal(0, std, (y.shape[0] - 2, y.shape[1]))
        np.savetxt(os.path.join(savename, "fitzhugh_nagumo_noise.dat"), np.hstack((t, y)))

    # Train
    var_list = pinn(t, y, noise, savename)

    # Prediction
    y = fitzhugh_nagumo_model(np.ravel(t), *var_list)
    np.savetxt(os.path.join(savename, "fitzhugh_nagumo_pred.dat"), np.hstack((t, y)))



if __name__ == "__main__":
    main()
