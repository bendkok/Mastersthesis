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
            pi_vals.append(pra_data["lowerbound"])
        
    
    inf_vals = np.array(inf_vals)
    re_vals = np.array(re_vals)
    pi_vals = np.array(pi_vals)
    
    return inf_vals, re_vals, pi_vals, tra_para
    

def pros_data(stat = '0', para = 'abt', noises=[0,1,2,5,10]):
    
    # noises = [0,1,2,5,10]
    inf_vals = []
    re_vals = []
    pi_vals = []
    
    
    for n in noises:
        
        inf_val, re_val, pi_val, tra_para = get_data(Path("fhn_res_clus/fhn_res_s-{}_v-{}_n{}_e40".format(stat, para, n)))
        
        inf_vals.append(inf_val)
        re_vals.append(re_val)
        pi_vals.append(pi_val)
    
    inf_vals = np.array(inf_vals)
    re_vals = np.array(re_vals)
    pi_vals = np.array(pi_vals)
    
    return inf_vals, re_vals, pi_vals, tra_para 


def run_make_plots(stat = '0', para = 'abt', save=False, noises=[0,1,2,5,10], do_scatter=False):
    
    inf_vals, re_vals, pi_vals, tra_para = pros_data(stat, para, noises=noises)
        
    make_inf_plot(inf_vals, tra_para, stat = stat, para = para, save=save, noises=noises)
    make_re_plot(re_vals, tra_para, stat = stat, para = para, save=save, noises=noises, do_scatter=do_scatter)
    make_pi_plot(pi_vals, tra_para, stat = stat, para = para, save=save, noises=noises)

    
def make_inf_plot(inf_vals, tra_para, stat = '0', para = 'abt', save=False, noises=[0,1,2,5,10], non_loc_path=None):
    
    pos_paras=['a','b','τ','I_ext']
    
    
    
    # inf_vals = np.array(inf_vals[1:])
    # tmp = np.zeros((len(inf_vals), *inf_vals[1].shape))
    
    # for i in range(len(inf_vals)):
    #     tmp[i] = inf_vals[i]
    
    # inf_vals = tmp
    
    
    inf_mean = np.array([np.mean(inf_vals[i], axis=0) for i in range(len(inf_vals)) ])
    inf_std = np.array([np.std(inf_vals[i], axis=0) for i in range(len(inf_vals)) ])
    # inf_std = np.std(inf_vals, axis=1)
    # print(inf_vals.shape, inf_mean.shape, inf_std.shape, len(noises), stat, para)
    
    noises_str = ['{}%'.format(no) for no in noises]
    
    grid_vals = np.arange(len(inf_mean), dtype=float)
    
    
    width = np.min(np.diff(grid_vals))/(inf_mean.shape[1]+1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
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
    ax.legend(loc='best', ncol=inf_mean.shape[1])
    
    if save:
        if non_loc_path==None:
            plt.savefig("bar_plots/s-{}_v-{}_inf.pdf".format(stat, para), bbox_inches='tight')
        else:
            plt.savefig(non_loc_path, bbox_inches='tight')
    plt.show()
    

def make_re_plot(re_vals, tra_para, stat = '0', para = 'abt', save=False, noises=[0,1,2,5,10], do_scatter=False, non_loc_path=None):
    
    pos_paras=['a','b','τ','I_ext']
    
    
    # print(re_vals)
    
    re_mean = np.array([np.mean(re_vals[i], axis=0) for i in range(len(re_vals)) ])
    re_std = np.array([np.std(re_vals[i], axis=0) for i in range(len(re_vals)) ])
    # re_mean = np.mean(re_vals, axis=1)
    # re_std = np.std(re_vals, axis=1)
    # print(re_vals.shape, re_mean.shape, re_std.shape, len(noises))
    
    do_plot = True
    for i in range(len(re_std)):
        for j in range(len(re_std[i])):
            if re_std[i,j] > re_mean[i,j]:
                print(stat, para, "{}%".format(noises[i]))
                print(re_vals[i][:,j])
                print(re_vals.shape, re_mean.shape, re_std.shape)
                do_plot=True
    
    
    # if do_plot and do_scatter:
    #     # print(re_mean.shape, re_vals.shape, [re_vals[i].shape for i in range(len(re_vals))])
    #     print(re_mean.shape)
    #     cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    #     for i in range(re_mean.shape[1]):    
    #         for j in range(re_mean.shape[0]):    
    #             # plt.scatter(range(len(re_vals[j,:,i])), re_vals[j,:,i].ravel(), c=cols[i])
    #             # plt.scatter(np.ones(len(re_vals[j,:,i].ravel()))*j + (i*.15), re_vals[j,:,i].ravel(), c=cols[i])
    #             plt.scatter(np.ones(len(re_vals[j][:,i].ravel()))*j + (i*.15), re_vals[j][:,i].ravel(), c=cols[i])
    #     plt.yscale('log')
    #     plt.show()
                
    
    noises_str = ['{}%'.format(no) for no in noises]
    
    grid_vals = np.arange(len(re_mean), dtype=float)
        
    width = np.min(np.diff(grid_vals))/(re_mean.shape[1]+1)
    
    if do_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
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
            bars.append(ax.boxplot(grid_vals+(width*(b+multi)), re_mean[:,b], width, yerr=re_std[:,b], 
                               capsize=4, label=re_paras[b]))
        
        ax.set_xticks(grid_vals + width / 2)
        ax.set_xticklabels(noises_str)
        ax.set_xlabel('Noise level')
        ax.set_ylabel('Relative error')
        ax.set_yscale('log')
        ax.legend(loc='best', ncol=re_mean.shape[1])
        
        if save:
            if non_loc_path==None:
                plt.savefig("bar_plots/s-{}_v-{}_re.pdf".format(stat, para), bbox_inches='tight')
            else:
                plt.savefig(non_loc_path, bbox_inches='tight')
        plt.show()



def make_pi_plot(pi_vals, tra_para, stat = '0', para = 'abt', save=False, noises=[0,1,2,5,10], non_loc_path=None):
    
    pos_paras=['a','b','τ','I_ext']
    
    
    # print(re_vals)
    
    pi_mean = np.array([np.mean(pi_vals[i], axis=0) for i in range(len(pi_vals)) ])
    pi_std = np.array([np.std(pi_vals[i], axis=0) for i in range(len(pi_vals)) ])
    # pi_mean = np.mean(pi_vals, axis=1)
    # pi_std = np.std(pi_vals, axis=1)
    # print(pi_vals.shape, pi_mean.shape, pi_std.shape, len(noises))
    
    noises_str = ['{}%'.format(no) for no in noises]
    
    grid_vals = np.arange(len(pi_mean), dtype=float)
        
    width = np.min(np.diff(grid_vals))/(pi_mean.shape[1]+1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
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
    ax.legend(loc='best', ncol=pi_mean.shape[1])
    
    if save:
        if non_loc_path==None:
            plt.savefig("bar_plots/s-{}_v-{}_pi.pdf".format(stat, para), bbox_inches='tight')
        else:
            plt.savefig(non_loc_path, bbox_inches='tight')
    plt.show()
    
    
    

def main():
    
    run_make_plots(stat = '0', para = 'b', save=True, do_scatter=True)
    run_make_plots(stat = '0', para = 'bt', save=True, do_scatter=True)
    run_make_plots(stat = '0', para = 'abt', save=True, do_scatter=True)
    # run_make_plots(stat = '0', para = 'abtI', save=True, noises=[0])
    run_make_plots(stat = '01', para = 'a', save=True, do_scatter=True)
    run_make_plots(stat = '01', para = 'ab', save=True, do_scatter=True)
    run_make_plots(stat = '01', para = 'abtI', save=True, do_scatter=True)

if __name__ == "__main__":
    main()




