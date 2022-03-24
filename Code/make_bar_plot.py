# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:30:24 2022

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import os
import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

dir_name = os.listdir('./fhn_res/fhn_res_clus')
pos_para = ["a", "b", "τ", "I"]
pos_states = ["v", "w"]
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


df = pd.DataFrame(dirs)
observed_states = df['observed_states'].to_numpy()
states = []
for s in observed_states:
    states.append(np.array(pos_states)[s])
df["States"] = states

var_trainable = df['var_trainable'].to_numpy()
var = []
for v in var_trainable:
    var.append(np.array(pos_para)[v])
df["Fitted"] = var


df = df.rename(columns={'noise':'Noise', 'param_mre':'MRE', 'sec_num_epochs':'Epochs'}, errors="raise")
df = df.sort_values(by=['States', 'Fitted', 'Noise', 'MRE'], ascending=[True,False,True,False], key=lambda col: col.astype(str).str.lower())


df['States'] = df['States'].apply(tuple)
df['Fitted'] = df['Fitted'].apply(tuple)


df1 = df.drop_duplicates(subset=['States', 'Fitted', 'Noise'], keep='last')

# print(df1[['States', 'Fitted', 'Noise', 'MRE', 'Epochs']])
sns.set_theme()

def make_plot_noise(states, fits, df_input, plot_re=True, plot_epochs=True, plot_found=False, plot_runtime=True):

    fits_loc = [ np.where(l == np.array(pos_para))[0] for l in fits]
    
    df_plot = df_input[(df_input['States'] == states) & ((df['Fitted'] == fits))]
    
    para_re = np.array(df_plot['param_re'].to_numpy()) #.iloc[0]
    para_re = np.array([element for element in para_re])
    
    found_param = df_plot['found_param'].to_numpy()
    found_param = np.array([element for element in found_param])
    
    # print(found_param)
    for i in range(len(fits)):
        df_plot = df_plot.assign(tmp0 = para_re[:,fits_loc[i]]).rename(columns={'tmp0':'RE {}'.format(fits[i])}, errors="raise")
        df_plot = df_plot.assign(tmp1 = found_param[:,i]).rename(columns={'tmp1':'{}'.format(fits[i])}, errors="raise")
    
    rela = ["RE {}".format(i) for i in fits]
    found = ["{}".format(i) for i in fits]
    noise_plt = df_plot['Noise'].to_numpy()
    noise_plt = ['{:2.0%}'.format(n) for n in noise_plt]
    df_plot['Noise'] = noise_plt
    
    # ax = df_plot[['Noise', 'States', 'Fitted', 'MRE', 'RE a', 'RE b']].set_index('Noise').plot.bar(rot=0)
    if plot_re:
        ax = df_plot[['Noise', 'States', 'Fitted', 'MRE', *rela]].set_index('Noise').plot.bar(rot=0)
        ax.set_ylabel("Error")
        ax.set_yscale('log')
        title = "Relative error with known states {}, and fitted parameters {}".format(', '.join([s for s in states]), ', '.join([f for f in fits]))
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig("bar_plots/bar_re_noise_{}_{}.pdf".format(''.join([s for s in states]), ''.join([f for f in fits])))
    
    if plot_epochs:
        ax = df_plot[['Noise', 'States', 'Fitted', 'Epochs']].set_index('Noise').plot.bar(rot=0)
        ax.set_ylabel("Steps")
        title = "Epochs with known states {}, and fitted parameters {}".format(', '.join([s for s in states]), ', '.join([f for f in fits]))
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig("bar_plots/epoch/bar_epoch_noise_{}_{}.pdf".format(''.join([s for s in states]), ''.join([f for f in fits])))
    
    if plot_found:
        ax = df_plot[['Noise', 'States', 'Fitted', *found]].set_index('Noise').plot.bar(rot=0)
        ax.set_yscale("symlog")
        title = "Found parameters with known states {}, and fitted parameters {}".format(', '.join([s for s in states]), ', '.join([f for f in fits]))
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig("bar_plots/found/bar_found_noise_{}_{}.pdf".format(''.join([s for s in states]), ''.join([f for f in fits])))
        
    if plot_runtime:
        ax = df_plot[['Noise', 'States', 'Fitted', 'runtime']].set_index('Noise').plot.bar(rot=0)
        title = "Runtime with known states {}, and fitted parameters {}".format(', '.join([s for s in states]), ', '.join([f for f in fits]))
        ax.set_ylabel("Seconds")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig("bar_plots/runtime/bar_rt_noise_{}_{}.pdf".format(''.join([s for s in states]), ''.join([f for f in fits])))



def make_plot_experiment(noise, df_input, plot_re=True, plot_epochs=True, plot_found=False, plot_runtime=True):

    
    df_plot = df_input[df_input['Noise'] == noise]
    
    expe = []
    # print(str(df_plot['States'].iloc[0]))
    for i in range(df_plot.shape[0]):
        expe.append( ','.join([s for s in df_plot['States'].iloc[i]]) + ' ' + ','.join([f for f in df_plot['Fitted'].iloc[i]]))
    df_plot['Experiment'] = expe # df_plot['States'].astype(str) + df_plot['Fitted'].astype(str)
    # print(df_plot['Experiment'])
    
    fits_loc = []
    for o in df_plot['Fitted'].to_numpy():
        fits_loc.append( np.array([ np.where(l == np.array(pos_para))[0][0] for l in o]) )
    # print(fits_loc)
    
    para_re = np.array(df_plot['param_re'].to_numpy()) #.iloc[0]
    para_re = np.array([element for element in para_re])
    # print(para_re)
    
    para_re0 = np.zeros((df_plot.shape[0], 4))
    para_re0[:,:] = None #[None]*df_plot.shape[0]*4
    # print(found_param0)
    
    for i in range(len(fits_loc)):
        para_re0[i,fits_loc[i]] = para_re[i,fits_loc[i]]
    
    # print(para_re0)
    
    found_param = df_plot['found_param'].to_numpy()
    found_param = np.array([element for element in found_param])
    
    found_param0 = np.zeros((df_plot.shape[0], 4))
    found_param0[:,:] = None #[None]*df_plot.shape[0]*4
    # print(found_param0)
    
    for i in range(len(fits_loc)):
        found_param0[i,fits_loc[i]] = found_param[i]
    
    # print("found_param0", found_param0)
    
    for i in range(len(pos_para)):
        df_plot = df_plot.assign(tmp0 = para_re0[:,i]).rename(columns={'tmp0':'RE {}'.format(pos_para[i])}, errors="raise")
        df_plot = df_plot.assign(tmp1 = found_param0[:,i]).rename(columns={'tmp1':'{}'.format(pos_para[i])}, errors="raise")
    
    rela = ["RE {}".format(i) for i in pos_para]
    found = ["{}".format(i) for i in pos_para]
    
    if plot_re:
        ax = df_plot[['Experiment', 'MRE', *rela]].set_index('Experiment').plot.bar(rot=0)
        ax.set_ylabel("Error")
        ax.set_yscale('log')
        title = "Relative error with noise = {:>3.0%}".format(noise)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig("bar_plots/bar_re_expe_{}.pdf".format(int(noise*100)))
        
    if plot_epochs:
        ax = df_plot[['Experiment', 'Epochs']].set_index('Experiment').plot.bar(rot=0)
        # ax = df_plot[['Noise', 'States', 'Fitted', 'Epochs']].set_index('Noise').plot.bar(rot=0)
        ax.set_ylabel("Steps")
        title = "Epochs with noise = {:>3.0%}".format(noise)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig("bar_plots/epoch/bar_epoch_expe_{}.pdf".format(int(noise*100)))
    
    if plot_found:
        ax = df_plot[['Experiment', *found]].set_index('Experiment').plot.bar(rot=0)
        # ax = df_plot[['Noise', 'States', 'Fitted', *found]].set_index('Noise').plot.bar(rot=0)
        ax.set_yscale("symlog")
        title = "Found parameters with noise = {:>3.0%}".format(noise)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig("bar_plots/found/bar_found_expe_{}.pdf".format(int(noise*100)))
        
    if plot_runtime:
        ax = df_plot[['Experiment', 'runtime']].set_index('Experiment').plot.bar(rot=0)
        # ax = df_plot[['Noise', 'States', 'Fitted', 'runtime']].set_index('Noise').plot.bar(rot=0)
        title = "Runtime with noise = {:>3.0%}".format(noise)
        ax.set_ylabel("Seconds")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig("bar_plots/runtime/bar_rt_expe_{}.pdf".format(int(noise*100)))
    

expe = df1[['States', 'Fitted']].drop_duplicates().to_numpy()
print(expe)

for i in range(len(expe)):
    make_plot_noise(*expe[i], df1, plot_found=True)
    # make_plot_noise(*expe[i], df1, plot_epochs=False, plot_found=False, plot_runtime=False)


# noises = [.0, .01, .02, .05, .10]
noises = df1['Noise'].drop_duplicates().to_numpy()
print(noises)
for n in noises:    
    make_plot_experiment(n, df1, plot_found=True)



