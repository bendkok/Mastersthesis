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
from datetime import timedelta
from time import gmtime
from time import strftime

sns.set_theme()
import warnings
warnings.filterwarnings("ignore")

def import_data(dir_name = os.listdir('./fhn_res/fhn_res_clus')):
    
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
            with open(os.path.join(path, "pra_ident.pkl"), "rb") as a_file:
                par_data = pickle.load(a_file)
            data = eva_data.copy()
            data.update(hyp_data)
            data.update(par_data)
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

    return df1



def make_plot_noise(states, fits, df_input, plot_re=True, plot_epochs=True, plot_found=False, plot_runtime=True):
    
    pos_para = ["a", "b", "τ", "I"]
    pos_states = ["v", "w"]
    
    fits_loc = [ np.where(l == np.array(pos_para))[0] for l in fits]
    
    df_plot = df_input[(df_input['States'] == states) & ((df_input['Fitted'] == fits))]
    
    para_re = np.array(df_plot['param_re'].to_numpy()) #.iloc[0]
    para_re = np.array([element for element in para_re])
    
    found_param = df_plot['found_param'].to_numpy()
    found_param = np.array([element for element in found_param])
    
    re_err = np.array(df_plot['err'].to_numpy()) #.iloc[0]
    re_err = np.array([element for element in re_err])
    
    found_err = df_plot['lowerbound'].to_numpy()
    found_err = np.array([element for element in found_err])
    
    df_error = df_plot[['Noise']]
    
    # print(found_param)
    for i in range(len(fits)):
        df_plot = df_plot.assign(tmp0 = para_re[:,fits_loc[i]]).rename(columns={'tmp0':'RE {}'.format(fits[i])}, errors="raise")
        df_plot = df_plot.assign(tmp1 = found_param[:,i]).rename(columns={'tmp1':'{}'.format(fits[i])}, errors="raise")
        
        df_error = df_error.assign(tmp2 = found_err[:,i]).rename(columns={'tmp2':'RE {}'.format(fits[i])}, errors="raise")
        df_error = df_error.assign(tmp3 = found_err[:,i]).rename(columns={'tmp3':'{}'.format(fits[i])}, errors="raise")
    
    
    rela = ["RE {}".format(i) for i in fits]
    found = ["{}".format(i) for i in fits]
    noise_plt = df_plot['Noise'].to_numpy()
    noise_plt = ['{:2.0%}'.format(n) for n in noise_plt]
    df_plot['Noise'] = noise_plt
    df_error['Noise'] = noise_plt
    
    # ax = df_plot[['Noise', 'States', 'Fitted', 'MRE', 'RE a', 'RE b']].set_index('Noise').plot.bar(rot=0)
    if plot_re:
        if len(rela)>1:
            ax = df_plot[['Noise', 'States', 'Fitted', 'MRE', *rela]].set_index('Noise').plot.bar(rot=0, width=.8)
            rotation=90
        else:
            ax = df_plot[['Noise', 'States', 'Fitted', *rela]].set_index('Noise').plot.bar(rot=0, width=.8)
            rotation=0
        ax.set_ylabel("Error")
        ax.set_yscale('log')
        
        if len(df_plot['Noise'])==1:
            rotation=0
        for bars in ax.containers:
            ax.bar_label(bars, fmt='%.2e', rotation=rotation, size=8, padding=3, label_type='edge')
        ax.margins(y=0.3)
        
        title = "Relative error with known states {}, and fitted parameters {}".format(', '.join([s for s in states]), ', '.join([f for f in fits]))
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig("bar_plots_old/bar_re_noise_{}_{}.pdf".format(''.join([s for s in states]), ''.join([f for f in fits])))
    
    if plot_epochs:
        ax = df_plot[['Noise', 'States', 'Fitted', 'Epochs']].set_index('Noise').plot.bar(rot=0)
        ax.set_ylabel("Steps")
        title = "Epochs with known states {}, and fitted parameters {}".format(', '.join([s for s in states]), ', '.join([f for f in fits]))
        ax.set_title(title)
        
        for bars in ax.containers:
            ax.bar_label(bars, labels=[f'{x:,.0f}' for x in bars.datavalues], rotation=0, size=12, padding=3, label_type='edge')
        ax.margins(y=0.3)
        
        plt.tight_layout()
        plt.savefig("bar_plots_old/epoch/bar_epoch_noise_{}_{}.pdf".format(''.join([s for s in states]), ''.join([f for f in fits])))
    
    if plot_found:
        df_tmp = {'Noise':'Real', 'a':-0.3, 'b': 1.1, 'τ': 20, 'I': 0.23}
        df_plot0 = df_plot.append(df_tmp, ignore_index = True)
        df_error0 = df_error.append({'Noise':'Real','a':0.,'b':.0,'τ':.0,'I':0.}, ignore_index = True)
        ax = df_plot0[['Noise', *found]].set_index('Noise').plot.bar(yerr=df_error0[['Noise', *found]].set_index('Noise'), capsize=3,  rot=0, width=.8)
        if len(rela)>1:    
            ax.set_yscale("symlog")
        title = "Found parameters with known states {}, and fitted parameters {}".format(', '.join([s for s in states]), ', '.join([f for f in fits]))
        ax.set_title(title)
                
        if (len(rela)==4) and (len(df_plot['Noise'])>1):
            rotation=45
        else:
            rotation=0
        for bars in ax.containers[1::2]:
            ax.bar_label(bars, fmt='%.3f', rotation=rotation, size=8, padding=3, label_type='edge')
        ax.margins(y=0.2)
        
        plt.tight_layout()
        plt.savefig("bar_plots_old/found/bar_found_noise_{}_{}.pdf".format(''.join([s for s in states]), ''.join([f for f in fits])))
        
    if plot_runtime:
        ax = df_plot[['Noise', 'States', 'Fitted', 'runtime']].set_index('Noise').plot.bar(rot=0)
        title = "Runtime with known states {}, and fitted parameters {}".format(', '.join([s for s in states]), ', '.join([f for f in fits]))
        ax.set_ylabel("Seconds")
        ax.set_title(title)
        
        for bars in ax.containers:
            ax.bar_label(bars, labels=[strftime("%H:%M:%S", gmtime(x)) for x in bars.datavalues], rotation=0, size=12, padding=3, label_type='edge')
        ax.margins(y=0.3)
        
        plt.tight_layout()
        plt.savefig("bar_plots_old/runtime/bar_rt_noise_{}_{}.pdf".format(''.join([s for s in states]), ''.join([f for f in fits])))



def make_plot_experiment(noise, df_input, plot_re=True, plot_epochs=True, plot_found=False, plot_runtime=True):

    pos_para = ["a", "b", "τ", "I"]
    pos_states = ["v", "w"]
    
    
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
    
    found_err = df_plot['lowerbound'].to_numpy()
    found_err = np.array([element for element in found_err])
    
    found_err0 = np.zeros((df_plot.shape[0], 4))
    found_err0[:,:] = None #[None]*df_plot.shape[0]*4
    
    for i in range(len(fits_loc)):
        found_param0[i,fits_loc[i]] = found_param[i]
        found_err0[i,fits_loc[i]] = found_err[i]
    
    # print("found_param0", found_param0)
    
    df_error = df_plot[['Experiment']]
    
    for i in range(len(pos_para)):
        df_plot = df_plot.assign(tmp0 = para_re0[:,i]).rename(columns={'tmp0':'RE {}'.format(pos_para[i])}, errors="raise")
        df_plot = df_plot.assign(tmp1 = found_param0[:,i]).rename(columns={'tmp1':'{}'.format(pos_para[i])}, errors="raise")
        
        df_error = df_error.assign(tmp2 = found_err0[:,i]).rename(columns={'tmp2':'{}'.format(pos_para[i])}, errors="raise")
    
    rela = ["RE {}".format(i) for i in pos_para]
    found = ["{}".format(i) for i in pos_para]
    
    
    if plot_re:
        ax = df_plot[['Experiment', 'MRE', *rela]].set_index('Experiment').plot.bar(rot=0, width=.8)
        ax.set_ylabel("Error")
        ax.set_yscale('log')
        title = "Relative error with noise = {:>3.0%}".format(noise)
        ax.set_title(title)
        for bars in ax.containers:
            ax.bar_label(bars, fmt='%.2e', rotation=90, size=8, padding=3)
        ax.margins(y=0.3)
        
        plt.tight_layout()
        plt.savefig("bar_plots_old/bar_re_expe_{}.pdf".format(int(noise*100)))
        
    if plot_epochs:
        ax = df_plot[['Experiment', 'Epochs']].set_index('Experiment').plot.bar(rot=0)
        # ax = df_plot[['Noise', 'States', 'Fitted', 'Epochs']].set_index('Noise').plot.bar(rot=0)
        ax.set_ylabel("Steps")
        title = "Epochs with noise = {:>3.0%}".format(noise)
        ax.set_title(title)
        for bars in ax.containers:
            ax.bar_label(bars, labels=[f'{x:,.0f}' for x in bars.datavalues], rotation=0, size=12, padding=3, label_type='edge')
        ax.margins(y=0.3)
        
        plt.tight_layout()
        plt.savefig("bar_plots_old/epoch/bar_epoch_expe_{}.pdf".format(int(noise*100)))
    
    if plot_found:
        df_tmp = {'Experiment':'Real', 'a':-0.3, 'b': 1.1, 'τ': 20, 'I': 0.23}
        df_plot0 = df_plot.append(df_tmp, ignore_index = True)
        # df_error0 = df_error.append({'Experiment':'Real','a':0.,'b':.0,'τ':.0,'I':0.}, ignore_index = True)
        ax = df_plot0[['Experiment', *found]].set_index('Experiment').plot.bar(yerr=df_error[['Experiment', *found]].set_index('Experiment'), capsize=2,  rot=0)
        ax.set_yscale("symlog")
        title = "Found parameters with noise = {:>3.0%}".format(noise)
        ax.set_title(title)
        
        # if (len(rela)==4) and (len(df_plot['Experiment'])>1):
        #     rotation=45
        # else:
        #     rotation=0
        
        # for bars in ax.containers[1::2]:
        #     # [print(a) for a in bars]
        #     # print(bars)
        #     ax.bar_label(bars, fmt='%.3f', rotation=90, size=8, padding=3, label_type='edge')
        # ax.margins(y=0.3)
        
        plt.tight_layout()
        plt.savefig("bar_plots_old/found/bar_found_expe_{}.pdf".format(int(noise*100)))
        
    if plot_runtime:
        ax = df_plot[['Experiment', 'runtime']].set_index('Experiment').plot.bar(rot=0)
        # ax = df_plot[['Noise', 'States', 'Fitted', 'runtime']].set_index('Noise').plot.bar(rot=0)
        title = "Runtime with noise = {:>3.0%}".format(noise)
        ax.set_ylabel("Seconds")
        ax.set_title(title)
        
        for bars in ax.containers:
            ax.bar_label(bars, labels=[strftime("%H:%M:%S", gmtime(x)) for x in bars.datavalues], rotation=0, size=12, padding=3, label_type='edge')
        ax.margins(y=0.3)
        
        plt.tight_layout()
        plt.savefig("bar_plots_old/runtime/bar_rt_expe_{}.pdf".format(int(noise*100)))

def main():
    
    df1 = import_data()
    
    expe = df1[['States', 'Fitted']].drop_duplicates().to_numpy()
    print(expe)
    
    for i in range(len(expe)):
        make_plot_noise(*expe[i], df1, plot_found=True, plot_epochs=True, plot_re=True, plot_runtime=True)
        # make_plot_noise(*expe[i], df1, plot_epochs=True, plot_found=True, plot_runtime=True)
    
    
    noises = df1['Noise'].drop_duplicates().to_numpy()
    print(noises)
    for n in noises:    
        make_plot_experiment(n, df1, plot_found=True, plot_epochs=True, plot_re=True, plot_runtime=True)

if __name__ == main():
    
    main()


