# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:31:26 2021

@author: benda
"""

import numpy as np
import sympy as sp
import sympy.calculus.util as ut
import matplotlib.pyplot as plt

a = -0.3
b = 1.1
tau = 20
Iext = 0.23
t = np.linspace(0, 999)
#v = np.linspace(-1.9304301691509158, 1.915302446875821, 10000)
v = sp.Symbol('v')
# print(v)

lam1 = ( -v**2 - (b/tau) + 1 + sp.sqrt( (1 - v**2 + b/tau)**2 + 4/tau ) )/2
lam2 = ( -v**2 - (b/tau) + 1 - sp.sqrt( (1 - v**2 + b/tau)**2 + 4/tau ) )/2

dlam1 = sp.diff(lam1, v)
dlam2 = sp.diff(lam2, v)


def get_exte(function, v, lower_bound, upper_bound):
    zeros = sp.solveset(function, v, domain=sp.Interval(lower_bound, upper_bound))
    assert zeros.is_FiniteSet # If there are infinite solutions the next line will hang.
    min1 = sp.Min(function.subs(v, lower_bound), function.subs(v, upper_bound), *[function.subs(v, i) for i in zeros])
    max1 = sp.Max(function.subs(v, lower_bound), function.subs(v, upper_bound), *[function.subs(v, i) for i in zeros])
    
    return min1, max1

lower_bound = -2
upper_bound = 2


min1, max1 = get_exte(lam1, v, lower_bound, upper_bound)
min2, max2 = get_exte(lam2, v, lower_bound, upper_bound)
print(min1, min2)
print(max1, max2)


S = np.abs(np.max([max1, max2]) / np.min([min1, min2]) )
print(S, S * (t[-1] - t[0]))

vs = np.linspace(-2, 2)

lam1s = [lam1.subs(v, i) for i in vs]
lam2s = [lam2.subs(v, i) for i in vs]

print("Maximum:", max(lam1s), max(lam2s))
print("Minimum:", min(lam1s), min(lam2s))
print("S: ", np.abs(max(lam1s) / min(lam2s)), np.abs(max(lam1s) / min(lam2s)) * (t[-1] - t[0]))

plt.plot(vs, lam1s, label="λ+")
plt.plot(vs, lam2s, label="λ-")
plt.xlabel("v")
plt.ylabel("λ")
plt.legend()
plt.savefig("lambda_plots.pdf")
plt.show()




