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


# infered value
# RE 
# pra_iden std
# runtime?

def get_data(path):
    
    inf_vals = []
    re_vals = []
    pi_vals = []
    
    paths = os.listdir(path)
    if len(paths)<10:
        print(path)
        # return None, None, None
    
    with open(os.path.join(os.path.join(path, paths[0]), "hyperparameters.pkl"), "rb") as a_file:
        hyp_data = pickle.load(a_file)
        tra_para = hyp_data["var_trainable"]
        
    for i in range(len(paths)):
        # cur_path = os.path.join(path, "expe_{}".format(i))
        cur_path = os.path.join(path, paths[i])
        
        with open(os.path.join(cur_path, "evaluation.pkl"), "rb") as a_file:
            eva_data = pickle.load(a_file)
            inf_vals.append(eva_data["found_param"])
            re_vals.append(eva_data["param_re"][np.where(tra_para)])
        
        with open(os.path.join(cur_path, "pra_ident.pkl"), "rb") as a_file:
            pra_data = pickle.load(a_file)
            # pi_vals.append(pra_data["err"])
            pi_vals.append(pra_data["lowerbound"])
        
    
    inf_vals = np.array(inf_vals)
    re_vals = np.array(re_vals)
    pi_vals = np.array(pi_vals)
    
    return inf_vals, re_vals, pi_vals, tra_para
    

def pros_data(stat = '0', para = 'abt', noises=[0,1,2,5,10], epe=40, lr=""):
    
    # noises = [0,1,2,5,10]
    inf_vals = []
    re_vals = []
    pi_vals = []
    
    
    for n in noises:
        
        if lr == "0":
            inf_val, re_val, pi_val, tra_para = get_data(Path("fhn_res_clus_lr/fhn_res_s-{}_v-{}_n{}_e{}".format(stat, para, n, epe)))
            # print(lr)
        else:
            inf_val, re_val, pi_val, tra_para = get_data(Path("fhn_res_clus/fhn_res_s-{}_v-{}_n{}_e{}".format(stat, para, n, epe)))
            # print("Not lr")
        
        inf_vals.append(inf_val)
        re_vals.append(re_val)
        pi_vals.append(pi_val)
    
    inf_vals = np.array(inf_vals)
    re_vals = np.array(re_vals)
    pi_vals = np.array(pi_vals)
    
    return inf_vals, re_vals, pi_vals, tra_para 


def run_make_plots(stat = '0', para = 'abt', save=False, noises=[0,1,2,5,10], do_scatter=False, epe=40, lr=""):
    
    inf_vals, re_vals, pi_vals, tra_para = pros_data(stat, para, noises=noises, epe=epe, lr=lr)
        
    make_inf_plot(inf_vals, tra_para, stat = stat, para = para, save=save, noises=noises, do_scatter=do_scatter, epe=epe, lr=lr)
    
    make_re_plot(re_vals, tra_para, stat = stat, para = para, save=save, noises=noises, do_scatter=do_scatter, epe=epe, lr=lr)
    
    make_pi_plot(pi_vals, tra_para, stat = stat, para = para, save=save, noises=noises, do_scatter=do_scatter, epe=epe, lr=lr)

    
def make_inf_plot(inf_vals, tra_para, stat = '0', para = 'abt', save=False, noises=[0,1,2,5,10], non_loc_path=None, do_scatter=False, epe=40, lr=""):
    
    pos_paras=['a','b','τ','I_ext']
    
    
    inf_mean = np.array([np.mean(inf_vals[i], axis=0) for i in range(len(inf_vals)) ])
    inf_std = np.array([np.std(inf_vals[i], axis=0) for i in range(len(inf_vals)) ])
    
    noises_str = ['{}%'.format(no) for no in noises]
    
    if do_scatter:
        cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        for i in range(inf_mean.shape[1]):    
            for j in range(inf_mean.shape[0]):    
                plt.scatter(np.ones(len(inf_vals[j][:,i].ravel()))*j + (i*.15), inf_vals[j][:,i].ravel(), c=cols[i])
        # plt.yscale('symlog')
        if inf_mean.shape[1]>1:    
            plt.yscale('symlog')
        plt.show()
    
    
    grid_vals = np.arange(len(inf_mean), dtype=float)
    width = np.min(np.diff(grid_vals))/(inf_mean.shape[1]+1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
        
    
    bars = []
    inf_paras = np.array(pos_paras)[np.where(tra_para)[0]]
    
    if inf_mean.shape[1] == 1:
        multi = .5
    elif inf_mean.shape[1] == 2 :
        multi = 0
    elif inf_mean.shape[1] == 3 :
        multi = -.5
    elif inf_mean.shape[1] == 4 :
        multi = -1
    
    for b in range(inf_mean.shape[1]):        
        bars.append(ax.bar(grid_vals+(width*(b+multi)), inf_mean[:,b], width, yerr=inf_std[:,b], 
                           capsize=4, label=inf_paras[b]))
    
    
    ax.set_xticks(grid_vals + width / 2)
    ax.set_xticklabels(noises_str)
    ax.set_xlabel('Noise level')
    ax.set_ylabel('Inferred value')
    if inf_mean.shape[1]>1:    
        ax.set_yscale('symlog')
    ax.legend(loc='best', ncol=inf_mean.shape[1], fontsize=14)
    
    if save:
        if non_loc_path==None:
            plt.savefig("bar_plots/s-{}_v-{}_e-{}_inf{}.pdf".format(stat, para, epe, lr), bbox_inches='tight')
        else:
            plt.savefig(non_loc_path, bbox_inches='tight')
    plt.show()
    

def make_re_plot(re_vals, tra_para, stat = '0', para = 'abt', save=False, noises=[0,1,2,5,10], do_scatter=False, non_loc_path=None, epe=40, lr=""):
    
    pos_paras=['a','b','τ','I_ext']

    re_mean = np.array([np.mean(re_vals[i], axis=0) for i in range(len(re_vals)) ])
    re_std = np.array([np.std(re_vals[i], axis=0) for i in range(len(re_vals)) ])

    
    do_plot = True
    # for i in range(len(re_std)):
    #     for j in range(len(re_std[i])):
    #         if re_std[i,j] > re_mean[i,j]:
    #             print(stat, para, "{}%".format(noises[i]))
    #             print(re_vals[i][:,j])
    #             print(re_vals.shape, re_mean.shape, re_std.shape)
    #             do_plot=True
    
    
    if do_plot and do_scatter:
        cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        for i in range(re_mean.shape[1]):    
            for j in range(re_mean.shape[0]):    
                plt.scatter(np.ones(len(re_vals[j][:,i].ravel()))*j + (i*.15), re_vals[j][:,i].ravel(), c=cols[i])
        plt.yscale('log')
        plt.show()
                
    
    noises_str = ['{}%'.format(no) for no in noises]
    
    grid_vals = np.arange(len(re_mean), dtype=float)
        
    width = np.min(np.diff(grid_vals))/(re_mean.shape[1]+1)
    
    if do_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(14)
        
        bars = []
        re_paras = np.array(pos_paras)[np.where(tra_para)[0]]
        
        if re_mean.shape[1] == 1:
            multi = .5
        elif re_mean.shape[1] == 2 :
            multi = 0
        elif re_mean.shape[1] == 3 :
            multi = -.5
        elif re_mean.shape[1] == 4 :
            multi = -1
        
        for b in range(re_mean.shape[1]):    
            bars.append(ax.bar(grid_vals+(width*(b+multi)), re_mean[:,b], width, yerr=re_std[:,b], 
                               capsize=4, label=re_paras[b]))
        
        ax.set_xticks(grid_vals + width / 2)
        ax.set_xticklabels(noises_str)
        ax.set_xlabel('Noise level')
        ax.set_ylabel('Relative error')
        ax.set_yscale('log')
        ax.legend(loc='best', ncol=re_mean.shape[1], fontsize=14)
        
        if save:
            if non_loc_path==None:
                plt.savefig("bar_plots/s-{}_v-{}_e-{}_re{}.pdf".format(stat, para, epe, lr), bbox_inches='tight')
            else:
                plt.savefig(non_loc_path, bbox_inches='tight')
        plt.show()



def make_pi_plot(pi_vals, tra_para, stat = '0', para = 'abt', save=False, noises=[0,1,2,5,10], non_loc_path=None, do_scatter=False, epe=40, lr=""):
    
    pos_paras=['a','b','τ','I_ext']
    
    # print(pi_vals.shape, [pi_vals[i].shape for i in range(5)])
    
    pi_mean = np.array([np.mean(pi_vals[i], axis=0) for i in range(len(pi_vals)) ])
    pi_std = np.array([np.std(pi_vals[i], axis=0) for i in range(len(pi_vals)) ])
    
    for i in range(pi_vals.shape[0]):
        for j in range(pi_vals.shape[2]):
            pi_curr = pi_vals[i,:,j]
            pi_curr = pi_curr[~np.isinf(pi_curr)]
            pi_mean[i,j] = np.mean(pi_curr)
            pi_std[i,j] = np.std(pi_curr)
    
    # print(pi_mean.shape)
    # print(pi_mean)
    
    noises_str = ['{}%'.format(no) for no in noises]
    
    if do_scatter:
        cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        for i in range(pi_mean.shape[1]):    
            for j in range(pi_mean.shape[0]):    
                plt.scatter(np.ones(len(pi_vals[j][:,i].ravel()))*j + (i*.15), pi_vals[j][:,i].ravel(), c=cols[i])
        plt.yscale('log')
        plt.show()
        
        
    
    grid_vals = np.arange(len(pi_mean), dtype=float)
    width = np.min(np.diff(grid_vals))/(pi_mean.shape[1]+1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    
    bars = []
    pi_paras = np.array(pos_paras)[np.where(tra_para)[0]]
    
    if pi_mean.shape[1] == 1:
        multi = .5
    elif pi_mean.shape[1] == 2 :
        multi = 0
    elif pi_mean.shape[1] == 3 :
        multi = -.5
    elif pi_mean.shape[1] == 4 :
        multi = -1
    
    for b in range(pi_mean.shape[1]):    
        bars.append(ax.bar(grid_vals+(width*(b+multi)), pi_mean[:,b], width, yerr=pi_std[:,b], 
                           capsize=4, label=pi_paras[b]))
    
    ax.set_xticks(grid_vals + width / 2)
    ax.set_xticklabels(noises_str)
    ax.set_xlabel('Noise level')
    ax.set_ylabel('Lowerbound STD')
    ax.set_yscale('log')
    ax.legend(loc='best', ncol=pi_mean.shape[1], fontsize=14)
    
    if save:
        if non_loc_path==None:
            plt.savefig("bar_plots/s-{}_v-{}_e-{}_pi{}.pdf".format(stat, para, epe, lr), bbox_inches='tight')
        else:
            plt.savefig(non_loc_path, bbox_inches='tight')
    plt.show()
    
    # print(pi_mean)
    
    

def main():
    
    # run_make_plots(stat = '0', para = 'b', save=True, do_scatter=True)
    # run_make_plots(stat = '0', para = 'bt', save=True, do_scatter=True)
    # run_make_plots(stat = '0', para = 'abt', save=True, do_scatter=True)
    
    # run_make_plots(stat = '01', para = 'a', save=True, do_scatter=True)
    # run_make_plots(stat = '01', para = 'ab', save=True, do_scatter=True)
    # run_make_plots(stat = '01', para = 'abtI', save=True, do_scatter=True)
    # run_make_plots(stat = '01', para = 'abtI', save=True, do_scatter=True, epe=80, lr="0")
    run_make_plots(stat = '1', para = 'abtI', save=True, do_scatter=False, epe=80)
    
    # run_make_plots(stat = '0', para = 'abtI', save=True, noises=[0])

if __name__ == "__main__":
    main()




