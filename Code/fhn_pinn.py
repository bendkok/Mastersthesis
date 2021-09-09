# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:13:48 2021

@author: benda
"""

import numpy as np
from scipy.integrate import odeint

import deepxde as dde
from deepxde.backend import tf

from glycosis import pinn
from fitzhugh_nagumo import fitzhugh_nagumo


def main():
    t = np.arange(0, 10, 0.005)[:, None]
    noise = 0.
    
    savename = "fitzhugh_nagumo_res/"

    # Data
    y = fitzhugh_nagumo(np.ravel(t))
    np.savetxt(savename+"fitzhugh_nagumo.dat", np.hstack((t, y)))
    # Add noise
    if noise > 0:
        std = noise * y.std(0)
        y[1:-1, :] += np.random.normal(0, std, (y.shape[0] - 2, y.shape[1]))
        np.savetxt(savename+"fitzhugh_nagumo_noise.dat", np.hstack((t, y)))

    # Train
    var_list = pinn(t, y, noise, savename)

    # Prediction
    y = fitzhugh_nagumo(np.ravel(t), *var_list)
    np.savetxt(savename+"fitzhugh_nagumo_pred.dat", np.hstack((t, y)))


if __name__ == "__main__":
    main()
