# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 21:33:24 2022

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


def get_data(path):
    
    inf_vals = []
    re_vals = []
    pi_vals = []
    nn_vals = []
    
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
            nn_vals.append(eva_data["nn_mse"])
        
        with open(os.path.join(cur_path, "pra_ident.pkl"), "rb") as a_file:
            pra_data = pickle.load(a_file)
            # pi_vals.append(pra_data["err"])
            pi_vals.append(pra_data["lowerbound"])
        
    
    inf_vals = np.array(inf_vals)
    re_vals = np.array(re_vals)
    pi_vals = np.array(pi_vals)
    nn_vals = np.array(nn_vals)
    
    return inf_vals, re_vals, pi_vals, nn_vals, tra_para
    

def pros_data(stat = '0', para = 'abt', noises=[0,1,2,5,10], epe=40, lr=""):
    
    # noises = [0,1,2,5,10]
    inf_vals = []
    re_vals = []
    pi_vals = []
    nn_vals = []
    
    
    for n in noises:
        
        if lr == "0":
            inf_val, re_val, pi_val, nn_val, tra_para = get_data(Path("fhn_res_clus_lr/fhn_res_s-{}_v-{}_n{}_e{}".format(stat, para, n, epe)))
            # print(lr)
        else:
            inf_val, re_val, pi_val, nn_val, tra_para = get_data(Path("fhn_res_clus/fhn_res_s-{}_v-{}_n{}_e{}".format(stat, para, n, epe)))
            # print("Not lr")
        
        inf_vals.append(inf_val)
        re_vals.append(re_val)
        pi_vals.append(pi_val)
        nn_vals.append(nn_val)
    
    inf_vals = np.array(inf_vals)
    re_vals = np.array(re_vals)
    # print(len(pi_vals), [pi_vals[l].shape for l in range(5)])
    pi_vals = np.array(pi_vals)
    nn_vals = np.array(nn_vals)
    
    return inf_vals, re_vals, pi_vals, nn_vals, tra_para 


def run_make_plots(stat = '0', para = 'abt', save=False, noises=[0,1,2,5,10], epe=40, lr=""):
    
    inf_vals, re_vals, pi_vals, nn_vals, tra_para = pros_data(stat, para, noises=noises, epe=epe, lr=lr)
        
    make_inf_plot(inf_vals, tra_para, stat=stat, para=para, save=save, noises=noises, epe=epe, lr=lr)
    
    make_re_plot(re_vals, tra_para, stat=stat, para=para, save=save, noises=noises, epe=epe, lr=lr)

    make_pi_plot(pi_vals, tra_para, stat=stat, para=para, save=save, noises=noises, epe=epe, lr=lr)
    
    make_nn_plot(nn_vals, tra_para, stat=stat, para=para, save=save, noises=noises, epe=epe, lr=lr)


def make_inf_plot(inf_vals, tra_para, stat = '0', para = 'abt', save=False, noises=[0,1,2,5,10], non_loc_path=None, epe=40, lr=""):
    
    pos_paras=['a','b','τ','I_ext']
        
    noises_str = ['{}%'.format(no) for no in noises]
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    
    def set_box_color(bp, color):
        for patch in bp['boxes']:
            patch.set_facecolor(color)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    boxes = []
    re_paras = np.array(pos_paras)[np.where(tra_para)[0]]
    
    grid_vals = np.arange(len(inf_vals), dtype=float)
    width = np.min(np.diff(grid_vals))/(inf_vals.shape[2]+1)
    
    if inf_vals.shape[2] == 1:
        multi = 0
    elif inf_vals.shape[2] == 2 :
        multi = -.5
    elif inf_vals.shape[2] == 3 :
        multi = -1
    elif inf_vals.shape[2] == 4 :
        multi = -1.5

    for b in range(inf_vals.shape[2]):    
        bplot = ax.boxplot(inf_vals[:,:,b].T,
                patch_artist=True,  # fill with color
                positions=grid_vals+(width*(b+multi)),
                widths=width,
                )  
        boxes.append(bplot)
    
    
    for b in range(inf_vals.shape[2]):
        set_box_color(boxes[b], colors[np.where(tra_para)[0][b]])
        plt.plot([], colors[np.where(tra_para)[0][b]], label=re_paras[b])
    plt.legend(fontsize=14)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    
    
    plt.xticks(range(0, len(noises_str), 1), noises_str)
    # ax.set_title('Rectangular box plot')
    ax.yaxis.grid(True)
    if inf_vals.shape[2] > 1:
        ax.set_yscale('symlog')
        # ax.set_yscale('log')
    # else:
    #     ax.set_yscale('symlog', linthresh=.1, linscale=.2)
    ax.set_xlabel('Noise level')
    ax.set_ylabel('Inferred value')

    if save:
        if non_loc_path==None:
            plt.savefig("box_plots/s-{}_v-{}_e{}_inf_box{}.pdf".format(stat, para, epe, lr), bbox_inches='tight')
        else:
            plt.savefig(non_loc_path, bbox_inches='tight')
    plt.show() 


def make_re_plot(re_vals, tra_para, stat = '0', para = 'abt', save=False, noises=[0,1,2,5,10], non_loc_path=None, epe=40, lr=""):
    
    pos_paras=['a','b','τ','I_ext']
        
    noises_str = ['{}%'.format(no) for no in noises]
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    
    def set_box_color(bp, color):
        for patch in bp['boxes']:
            patch.set_facecolor(color)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    boxes = []
    re_paras = np.array(pos_paras)[np.where(tra_para)[0]]
    
    grid_vals = np.arange(len(re_vals), dtype=float)
    width = np.min(np.diff(grid_vals))/(re_vals.shape[2]+1)
    
    if re_vals.shape[2] == 1:
        multi = 0
    elif re_vals.shape[2] == 2 :
        multi = -.5
    elif re_vals.shape[2] == 3 :
        multi = -1
    elif re_vals.shape[2] == 4 :
        multi = -1.5
    
    for b in range(re_vals.shape[2]):    
        bplot = ax.boxplot(re_vals[:,:,b].T,
                patch_artist=True,  # fill with color
                positions=grid_vals+(width*(b+multi)),
                widths=width,
                )  
        boxes.append(bplot)
    
    for b in range(re_vals.shape[2]):
        set_box_color(boxes[b], colors[np.where(tra_para)[0][b]])
        plt.plot([], colors[np.where(tra_para)[0][b]], label=re_paras[b])
    plt.legend(fontsize=14)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    
    plt.xticks(range(0, len(noises_str), 1), noises_str)
    # ax.set_title('Rectangular box plot')
    ax.yaxis.grid(True)
    ax.set_yscale('log')
    ax.set_xlabel('Noise level')
    ax.set_ylabel('Relative Error')
    
    
    if save:
        if non_loc_path==None:
            plt.savefig("box_plots/s-{}_v-{}_e{}_re_box{}.pdf".format(stat, para, epe, lr), bbox_inches='tight')
        else:
            plt.savefig(non_loc_path, bbox_inches='tight')
    plt.show()



def make_pi_plot(pi_vals, tra_para, stat = '0', para = 'abt', save=False, noises=[0,1,2,5,10], non_loc_path=None, epe=40, lr=""):
    
    pos_paras=['a','b','τ','I_ext']
        
    noises_str = ['{}%'.format(no) for no in noises]
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    
    def set_box_color(bp, color):
        for patch in bp['boxes']:
            patch.set_facecolor(color)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    boxes = []
    re_paras = np.array(pos_paras)[np.where(tra_para)[0]]
    
    grid_vals = np.arange(len(pi_vals), dtype=float)
    width = np.min(np.diff(grid_vals))/(pi_vals.shape[2]+1)
    
    if pi_vals.shape[2] == 1:
        multi = 0
    elif pi_vals.shape[2] == 2 :
        multi = -.5
    elif pi_vals.shape[2] == 3 :
        multi = -1
    elif pi_vals.shape[2] == 4 :
        multi = -1.5
    
    for b in range(pi_vals.shape[2]):    
        bplot = ax.boxplot(pi_vals[:,:,b].T,
                patch_artist=True,  # fill with color
                positions=grid_vals+(width*(b+multi)),
                widths=width,
                )  
        boxes.append(bplot)
    
    for b in range(pi_vals.shape[2]):
        set_box_color(boxes[b], colors[np.where(tra_para)[0][b]])
        plt.plot([], colors[np.where(tra_para)[0][b]], label=re_paras[b])
    plt.legend(fontsize=14)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    
    plt.xticks(range(0, len(noises_str), 1), noises_str)
    # ax.set_title('Rectangular box plot')
    ax.yaxis.grid(True)
    ax.set_yscale('log')
    ax.set_xlabel('Noise level')
    ax.set_ylabel('Lowerbound STD')
    
    if save:
        if non_loc_path==None:
            plt.savefig("box_plots/s-{}_v-{}_e{}_pi_box{}.pdf".format(stat, para, epe, lr), bbox_inches='tight')
        else:
            plt.savefig(non_loc_path, bbox_inches='tight')
    plt.show()
    
   


def make_nn_plot(nn_vals, tra_para, stat = '0', para = 'abt', save=False, noises=[0,1,2,5,10], non_loc_path=None, epe=40, lr=""):
    
    # pos_paras=['a','b','τ','I_ext']
        
    noises_str = ['{}%'.format(no) for no in noises]
    
    # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    
    def set_box_color(bp, color):
        for patch in bp['boxes']:
            patch.set_facecolor(color)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    boxes = []
    # re_paras = np.array(pos_paras)[np.where(tra_para)[0]]
    
    grid_vals = np.arange(len(nn_vals), dtype=float)
    width = np.min(np.diff(grid_vals))/(1+1)
    
    
    multi = 0

    for b in range(1):    
        bplot = ax.boxplot(nn_vals[:,:].T,
                patch_artist=True,  # fill with color
                positions=grid_vals+(width*(b+multi)),
                widths=width,
                )  
        boxes.append(bplot)

    # for b in range(1):
    #     set_box_color(boxes[b], colors[np.where(tra_para)[0][b]])
    #     plt.plot([], colors[np.where(tra_para)[0][b]], label=re_paras[b])
    # plt.legend(fontsize=14)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    
    plt.xticks(range(0, len(noises_str), 1), noises_str)
    # ax.set_title('Rectangular box plot')
    ax.yaxis.grid(True)
    ax.set_yscale('log')
    ax.set_xlabel('Noise level')
    ax.set_ylabel('MSE')

    if save:
        if non_loc_path==None:
            plt.savefig("box_plots/s-{}_v-{}_e{}_nn_box{}.pdf".format(stat, para, epe, lr), bbox_inches='tight')
        else:
            plt.savefig(non_loc_path, bbox_inches='tight')
    plt.show() 


    

def main():
    
    # run_make_plots(stat = '0', para = 'b', save=True)
    # run_make_plots(stat = '0', para = 'bt', save=True)
    # run_make_plots(stat = '0', para = 'abt', save=True)
    
    # run_make_plots(stat = '01', para = 'a', save=True)
    # run_make_plots(stat = '01', para = 'ab', save=True)
    # run_make_plots(stat = '01', para = 'abtI', save=True)
    # run_make_plots(stat = '01', para = 'abtI', save=True, epe=80, lr="0")
    run_make_plots(stat = '1', para = 'abtI', save=True, epe=80)
    
    # run_make_plots(stat = '0', para = 'abtI', save=True, noises=[0])

if __name__ == "__main__":
    main()




