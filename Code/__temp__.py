# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 18:06:54 2022

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import os

learning_rate = 1e-2
decay_rate = 1e2
decay_step = 2e4
global_step = np.linspace(0, 2e4, 1000)

decayed_learning_rate = learning_rate / (1 + decay_rate * global_step /
decay_step)


plt.plot(global_step, decayed_learning_rate)
plt.plot(global_step, [1e-2/(1+1e3)]*1000)

plt.yscale("log")

path = Path("fhn_res/fitzhugh_nagumo_res_a_12")

with open(os.path.join(path, "evaluation.pkl"), "rb") as a_file:
    data = pickle.load(a_file)

print(data)