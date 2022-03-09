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

# learning_rate = 1e-2
# decay_rate = 1e2
# decay_step = 1e4
# global_step = np.linspace(0, 1e4, 1000)

# decayed_learning_rate = learning_rate / (1 + decay_rate * global_step /
# decay_step)


# plt.plot(global_step, decayed_learning_rate)
# plt.plot(global_step, [1e-2/(1+1e2)]*1000)
# plt.yscale("log")
# plt.show()

# learning_rate = 1e-2
# decay_rate = .01
# decay_step = 1.
# global_step = np.linspace(0, 1e4, 1000)

# decayed_learning_rate = learning_rate / (1 + decay_rate * global_step /
# decay_step)


# plt.plot(global_step, decayed_learning_rate)
# plt.plot(global_step, [1e-2/(1+1e2)]*1000)
# plt.yscale("log")
# plt.show()

#display evaluation.pkl
path = Path("fhn_res/fitzhugh_nagumo_res_all_00")

with open(os.path.join(path, "evaluation.pkl"), "rb") as a_file:
    data = pickle.load(a_file)
print(data)

with open(os.path.join(path, "hyperparameters.pkl"), "rb") as a_file:
    hyp = pickle.load(a_file)
print(hyp)

#print losses from make_one_plot
parts = ['ODE ', 'BC  ', 'Data']


inp_dat = np.loadtxt(os.path.join(path, 'loss.dat'), delimiter=' ', skiprows=3, dtype=float)

epochs = inp_dat[:,0]
n_loss = inp_dat.shape[1]//2 +1
train_loss = inp_dat[:,1:n_loss]
test_loss = inp_dat[:,n_loss:]

print()
for i in range(len(parts)):
    diff = [np.min( np.abs( train_loss[:,i*2]/train_loss[:,i*2+1] ))]
    diff.append( np.mean( np.abs( train_loss[:,i*2]/train_loss[:,i*2+1] )) )
    diff.append( np.max( np.abs( train_loss[:,i*2]/train_loss[:,i*2+1] )) )
    # print("Mean loss difference {}: {}".format(parts[i], diff))
    print("Mean loss {}: {}, {}".format(parts[i], np.mean(train_loss[:,i*2]), np.mean(train_loss[:,i*2+1])))
print()
for i in range(len(parts)):
    print("Min  loss {}: {}, {}".format(parts[i], np.min(train_loss[:,i*2]), np.min(train_loss[:,i*2+1])))
    

diff = np.abs(train_loss[:2] - test_loss[:2])
print("\nODE train-test diff: ", np.mean(diff), np.min(diff), np.max(diff))





