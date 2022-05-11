# -*- coding: utf-8 -*-
"""
Created on Tue May 10 19:19:58 2022

@author: benda
"""

from fhn_pinn import fitzhugh_nagumo_model

from scipy.optimize import minimize, rosen, rosen_der
import numpy as np


x0 = [1.3, 0.7, 0.8, 1.9, 1.2]

res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)

print(res.x)

fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2

def fhn_fun(x):
    return fitzhugh_nagumo_model(np.linspace(0, 999, 1000), *x)    

res = minimize(fhn_fun, (-1,1,10,.1), method='Nelder-Mead')

print(res.x)
