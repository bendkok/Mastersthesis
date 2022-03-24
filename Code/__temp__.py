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
import pandas as pd

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
# path = Path("fhn_res/fitzhugh_nagumo_res_all_00")

# with open(os.path.join(path, "evaluation.pkl"), "rb") as a_file:
#     data = pickle.load(a_file)
# print(data)

# with open(os.path.join(path, "hyperparameters.pkl"), "rb") as a_file:
#     hyp = pickle.load(a_file)
# print(hyp)

# #print losses from make_one_plot
# parts = ['ODE ', 'BC  ', 'Data']


# inp_dat = np.loadtxt(os.path.join(path, 'loss.dat'), delimiter=' ', skiprows=3, dtype=float)

# epochs = inp_dat[:,0]
# n_loss = inp_dat.shape[1]//2 +1
# train_loss = inp_dat[:,1:n_loss]
# test_loss = inp_dat[:,n_loss:]

# print()
# for i in range(len(parts)):
#     diff = [np.min( np.abs( train_loss[:,i*2]/train_loss[:,i*2+1] ))]
#     diff.append( np.mean( np.abs( train_loss[:,i*2]/train_loss[:,i*2+1] )) )
#     diff.append( np.max( np.abs( train_loss[:,i*2]/train_loss[:,i*2+1] )) )
#     # print("Mean loss difference {}: {}".format(parts[i], diff))
#     print("Mean loss {}: {}, {}".format(parts[i], np.mean(train_loss[:,i*2]), np.mean(train_loss[:,i*2+1])))
# print()
# for i in range(len(parts)):
#     print("Min  loss {}: {}, {}".format(parts[i], np.min(train_loss[:,i*2]), np.min(train_loss[:,i*2+1])))
    

# diff = np.abs(train_loss[:2] - test_loss[:2])
# print("\nODE train-test diff: ", np.mean(diff), np.min(diff), np.max(diff))

dir_name = os.listdir('./fhn_res/fhn_res_clus')
pos_para = ["a", "b", "τ", "I"]
pos_states = ["v", "w"]
MREs = []
inc = []
dirs = []

for di in range(len(dir_name)):
    path = Path("fhn_res/fhn_res_clus/{}".format(dir_name[di]))
    try:
        with open(os.path.join(path, "evaluation.pkl"), "rb") as a_file:
            eva_data = pickle.load(a_file)
        with open(os.path.join(path, "hyperparameters.pkl"), "rb") as a_file:
            hyp_data = pickle.load(a_file)
        data = eva_data.copy()
        data.update(hyp_data)
        dirs.append(data)
    except:
        ""

# for i in range((len(dirs))):
#     curr_dir = dirs[i]
#     print("States: {0:>6}. Fitted Parameters: {1:>27}. Noise: {2:>3.0%}.".format(str(curr_dir['observed_states']), str(curr_dir['var_trainable']), curr_dir['noise']))
#     print("MRE: {:.5f}. Epochs: {:.1e}.\n".format(curr_dir['param_mre'], curr_dir['sec_num_epochs']))
#     # print(dirs[i]['param_mre'])


df = pd.DataFrame(dirs)
observed_states = df['observed_states'].to_numpy()
states = []
for s in observed_states:
    states.append(np.array(pos_states)[s])
    # states.append(np.array(pos_states)[s].tolist())
df["States"] = states
# print(df["States"])

var_trainable = df['var_trainable'].to_numpy()
var = []
for v in var_trainable:
    var.append(np.array(pos_para)[v])
df["Fitted"] = var
# print(df[["States", "Fitted"]])


df = df.rename(columns={'noise':'Noise', 'param_mre':'MRE', 'sec_num_epochs':'Epochs'}, errors="raise")
# print(df.States.str.len().sort_values())

# df = df.sort_values(by=['States', 'Fitted', 'Noise', 'Epochs'], ascending=[True,False,True,True], key=lambda col: col.astype(str).str.lower())
df = df.sort_values(by=['States', 'Fitted', 'Noise', 'MRE'], ascending=[True,False,True,False], key=lambda col: col.astype(str).str.lower())
df0 = df[['States', 'Fitted', 'Noise', 'MRE', 'Epochs']]

# print(df0)

df['States'] = df['States'].apply(tuple)
df['Fitted'] = df['Fitted'].apply(tuple)
# print(df0)


df1 = df.drop_duplicates(subset=['States', 'Fitted', 'Noise'], keep='last')

print(df[['States', 'Fitted', 'Noise', 'MRE', 'Epochs']])
print()
print(df1[['States', 'Fitted', 'Noise', 'MRE', 'Epochs']])
# print(df0.apply(tuple).duplicated())

# print(df0.shape, df1.shape, df.shape)

# print()
# print(df1[['States', 'Fitted', 'Noise', 'MRE', 'found_param']])



# print(df[['States', 'Fitted', 'Noise', 'MRE']].to_latex(index=False))


# print()
# print(df.value_counts(subset=['States', 'Fitted']))

# df2 = df1[(df1['States'] == ('v', 'w')) & ((df['Fitted'] == ('a',)))]
# # df2 = df1[(df1['Noise'] == 0)]

# print()
# print(df2[['States', 'Fitted', 'Noise', 'MRE', 'Epochs', 'param_re']])

# # print(type(df2['param_re'].iloc[0]))

# para_re = np.array(df2['param_re'].to_numpy()) #.iloc[0]
# para_re = np.array([element for (i,element) in enumerate(para_re)])

# print(para_re[:,0])

# df2['RE a'] = para_re[:,0]

# found_param = df2['found_param'].to_numpy()
# found_param = np.array([element for (i,element) in enumerate(found_param)])

# print("asdfa ", found_param[:,0])

# df2['Found a'] = found_param[:,0]

# ax = df2[['Noise', 'States', 'Fitted', 'MRE', 'RE a']].set_index('Noise').plot.bar(rot=0)
# ax = df2[['Noise', 'States', 'Fitted', 'Epochs']].set_index('Noise').plot.bar(rot=0)
# # ax = df2[['Noise', 'States', 'Fitted', 'Found a']].set_index('Noise').plot.bar(rot=0)












