# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 18:06:54 2022

@author: benda
"""

import numpy as np
from scipy.integrate import odeint
import deepxde as dde
from deepxde.backend import tf
import os
import time
import shutil
import matplotlib.pyplot as plt

import beeler_reuter.beeler_reuter_1977_version06 as br_model 

from postprocessing import saveplot
from make_plots import make_plots
from make_one_plot import make_one_plot, plot_losses


var_list = [] 

def ODE(t, y):
    # v1 = y[:, 0:1] - y[:, 0:1] ** 3 - y[:, 1:2] + var_list[3]
    # v2 = (y[:, 0:1] - var_list[0] - var_list[1] * y[:, 1:2]) / var_list[2]
    
    values = np.zeros((8,), dtype=np.float_)
    
    (
        E_Na,
        g_Na,
        g_Nac,
        g_s,
        IstimAmplitude,
        IstimEnd,
        IstimPeriod,
        IstimPulseDuration,
        IstimStart,
        C,
    ) = var_list

    # Expressions for the Sodium current component
    i_Na = (g_Nac + g_Na * (y[:, 0:1] * y[:, 0:1] * y[:, 0:1]) * y[:, 1:2] * y[:, 2:3]) * (-E_Na + y[:, 7:8])

    # Expressions for the m gate component
    alpha_m = (-47 - y[:, 7:8]) / (-1 + 0.009095277101695816 * np.exp(-0.1 * y[:, 7:8]))
    beta_m = 0.7095526727489909 * np.exp(-0.056 * y[:, 7:8])
    values[0] = (1 - y[:, 0:1]) * alpha_m - beta_m * y[:, 0:1]

    # Expressions for the h gate component
    alpha_h = 5.497962438709065e-10 * np.exp(-0.25 * y[:, 7:8])
    beta_h = 1.7 / (1 + 0.1580253208896478 * np.exp(-0.082 * y[:, 7:8]))
    values[1] = (1 - y[:, 1:2]) * alpha_h - beta_h * y[:, 1:2]

    # Expressions for the j gate component
    alpha_j = (
        1.8690473007222892e-10
        * np.exp(-0.25 * y[:, 7:8])
        / (1 + 1.6788275299956603e-07 * np.exp(-0.2 * y[:, 7:8]))
    )
    beta_j = 0.3 / (1 + 0.040762203978366204 * np.exp(-0.1 * y[:, 7:8]))
    values[2] = (1 - y[:, 2:3]) * alpha_j - beta_j * y[:, 2:3]

    # Expressions for the Slow inward current component
    E_s = -82.3 - 13.0287 * np.log(0.001 * y[:, 3:4])
    i_s = g_s * (-E_s + y[:, 7:8]) * y[:, 4:5] * y[:, 5:6]
    values[3] = 7.000000000000001e-06 - 0.07 * y[:, 3:4] - 0.01 * i_s

    # Expressions for the d gate component
    alpha_d = (
        0.095
        * np.exp(1 / 20 - y[:, 7:8] / 100)
        / (1 + 1.4332881385696572 * np.exp(-0.07199424046076314 * y[:, 7:8]))
    )
    beta_d = 0.07 * np.exp(-44 / 59 - y[:, 7:8] / 59) / (1 + np.exp(11 / 5 + y[:, 7:8] / 20))
    values[4] = (1 - y[:, 4:5]) * alpha_d - beta_d * y[:, 4:5]

    # Expressions for the f gate component
    alpha_f = (
        0.012
        * np.exp(-28 / 125 - y[:, 7:8] / 125)
        / (1 + 66.5465065250986 * np.exp(0.14992503748125938 * y[:, 7:8]))
    )
    beta_f = 0.0065 * np.exp(-3 / 5 - y[:, 7:8] / 50) / (1 + np.exp(-6 - y[:, 7:8] / 5))
    values[5] = (1 - y[:, 5:6]) * alpha_f - beta_f * y[:, 5:6]

    # Expressions for the Time dependent outward current component
    i_x1 = (
        0.0019727757115328517
        * (-1 + 21.75840239619708 * np.exp(0.04 * y[:, 7:8]))
        * np.exp(-0.04 * y[:, 7:8])
        * y[:, 6:7]
    )

    # Expressions for the X1 gate component
    alpha_x1 = (
        0.031158410986342627
        * np.exp(0.08264462809917356 * y[:, 7:8])
        / (1 + 17.41170806332765 * np.exp(0.05714285714285714 * y[:, 7:8]))
    )
    beta_x1 = (
        0.0003916464405623223
        * np.exp(-0.05998800239952009 * y[:, 7:8])
        / (1 + np.exp(-4 / 5 - y[:, 7:8] / 25))
    )
    values[6] = (1 - y[:, 6:7]) * alpha_x1 - beta_x1 * y[:, 6:7]

    # Expressions for the Time independent outward current component
    i_K1 = 0.0035 * (4.6000000000000005 + 0.2 * y[:, 7:8]) / (
        1 - 0.39851904108451414 * np.exp(-0.04 * y[:, 7:8])
    ) + 0.0035 * (-4 + 119.85640018958804 * np.exp(0.04 * y[:, 7:8])) / (
        8.331137487687693 * np.exp(0.04 * y[:, 7:8]) + 69.4078518387552 * np.exp(0.08 * y[:, 7:8])
    )

    # Expressions for the Stimulus protocol component
    Istim = (
        IstimAmplitude
        if t - IstimStart - IstimPeriod * np.floor((t - IstimStart) / IstimPeriod)
        <= IstimPulseDuration
        and t <= IstimEnd
        and t >= IstimStart
        else 0
    )
    # IstimAmplitude=0.5, IstimEnd=50000.0, IstimPeriod=1000.0, 
    # IstimPulseDuration=1.0, IstimStart=10.0


    # Expressions for the Membrane component
    values[7] = (-i_K1 - i_Na - i_s - i_x1 + Istim) / C

    # Return results
    return values
    
    
    # print(t.shape, t.shape.as_list(), tf.shape(y), y[0, :].shape.as_list()[0], len(var_list))
    # v = tf.transpose(tf.zeros_like(y[0]))
    # print(v.shape)
    # v = br_model.rhs(tf.transpose(t), tf.transpose(y), var_list)  
    # # for i in range(int(t.shape.as_list()[0])):
    # #     v[i] = br_model.rhs(t[i], y[i, :], var_list)  
    # res = []
    # for i in range(len(v)):
    #     res.append(tf.gradients(y[:, i:i+1], t)[0] - v[i],)
    # print(res)
    # return res
    
    