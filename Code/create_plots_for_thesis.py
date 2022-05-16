# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:59:30 2022

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt
from make_one_plot import make_comb_plot, plot_losses, make_comb_plot_v2
from pathlib import Path
# from make_bar_plot import pros_data, make_inf_plot, make_re_plot, make_pi_plot
import make_bar_plot as bar
# from make_box_plot import pros_data, make_inf_plot, make_re_plot, make_pi_plot
import make_box_plot as box
# from fitzhugh_nagumo import fitzhugh_nagumo
from fhn_pinn import fitzhugh_nagumo_model
from make_plots import make_plots
from out_dom import out_dom


#first figure
def first_good():
    path0 = Path("fhn_res_clus/fhn_res_s-01_v-a_n0_e40/expe_9")
    make_comb_plot(path0, non_loc_path="plots_for_thesis/comp_plot0.pdf")
    make_comb_plot_v2(path0, non_loc_path="plots_for_thesis/comp_plot0_v2.pdf")
    plot_losses(path0, non_loc_path="plots_for_thesis/loss_plot0.pdf")
    make_plots(path0, non_loc_path=Path("plots_for_thesis/epoch_chang"), params=[0], do_re_change=True, do_param_change=True)


def bad_figs():    
    #bad figure, wrong_k
    path1 = Path("fhn_res/fitzhugh_nagumo_res_feature_onlyb_2")
    make_comb_plot(path1, non_loc_path="plots_for_thesis/bad_res/wrong_k.pdf", do_tit=False)
    
    path2 = Path("fhn_res/fitzhugh_nagumo_res_feature_onlyb_6")
    make_comb_plot(path2, non_loc_path="plots_for_thesis/bad_res/corr_k.pdf", do_tit=False)
    
    path3 = Path("fhn_res/fitzhugh_nagumo_res_bas10_2_165")
    make_comb_plot(path3, non_loc_path="plots_for_thesis/bad_res/low_lr.pdf", do_tit=False)
    
    path4 = Path("fhn_res/fitzhugh_nagumo_res_a_21")
    make_comb_plot(path4, non_loc_path="plots_for_thesis/bad_res/high_lr.pdf", do_tit=False)
    
    path5 = Path("fhn_res/fitzhugh_nagumo_res_a_18")
    make_comb_plot(path5, non_loc_path="plots_for_thesis/bad_res/bad_lr_decay.pdf", do_tit=False)



#first box plots
def first_box():
    inf_vals, re_vals, pi_vals, nn_vals, tra_para = box.pros_data(stat='01', para='a')
    box.make_inf_plot(inf_vals, tra_para, stat='01', para='a', save=True, non_loc_path="plots_for_thesis/box0/box_inf0.pdf")
    box.make_re_plot(re_vals, tra_para, stat='01', para='a', save=True, non_loc_path="plots_for_thesis/box0/box_re0.pdf")
    box.make_pi_plot(pi_vals, tra_para, stat='01', para='a', save=True, non_loc_path="plots_for_thesis/box0/box_pi0.pdf")
    box.make_nn_plot(nn_vals, tra_para, stat='01', para='a', save=True, non_loc_path="plots_for_thesis/box0/box_nn0.pdf")


#second box plots
def sec_box():
    epe=80
    inf_vals, re_vals, pi_vals, nn_vals, tra_para = box.pros_data(stat='01', para='abtI', epe=epe)
    bar.make_inf_plot(inf_vals, tra_para, stat='01', para='abtI', save=True, non_loc_path="plots_for_thesis/box1/bar_inf1.pdf", epe=epe)
    box.make_re_plot(re_vals, tra_para, stat='01', para='abtI', save=True, non_loc_path="plots_for_thesis/box1/box_re1.pdf", epe=epe)
    bar.make_pi_plot(pi_vals, tra_para, stat='01', para='abtI', save=True, non_loc_path="plots_for_thesis/box1/bar_pi1.pdf", epe=epe)
    box.make_nn_plot(nn_vals, tra_para, stat='01', para='abtI', save=True, non_loc_path="plots_for_thesis/box1/box_nn1.pdf", epe=epe)
    
    
#third  box plots
def third_box():
    epe=40
    inf_vals, re_vals, pi_vals, nn_vals, tra_para = box.pros_data(stat='0', para='abt', epe=epe)
    bar.make_inf_plot(inf_vals, tra_para, stat='0', para='abt', save=True, non_loc_path="plots_for_thesis/box2/bar_inf2.pdf", epe=epe)
    box.make_re_plot(re_vals, tra_para, stat='0', para='abt', save=True, non_loc_path="plots_for_thesis/box2/box_re2.pdf", epe=epe)
    bar.make_pi_plot(pi_vals, tra_para, stat='0', para='abt', save=True, non_loc_path="plots_for_thesis/box2/bar_pi2.pdf", epe=epe)
    box.make_nn_plot(nn_vals, tra_para, stat='0', para='abt', save=True, non_loc_path="plots_for_thesis/box2/box_nn2.pdf", epe=epe)
    

#fourth box plots
def four_box():
    epe=100
    inf_vals, re_vals, pi_vals, nn_vals, tra_para = box.pros_data(stat='1', para='abtI', epe=epe)
    bar.make_inf_plot(inf_vals, tra_para, stat='1', para='abtI', save=True, non_loc_path="plots_for_thesis/box3/bar_inf3.pdf", epe=epe)
    box.make_re_plot(re_vals, tra_para, stat='1', para='abtI', save=True, non_loc_path="plots_for_thesis/box3/box_re3.pdf", epe=epe)
    bar.make_pi_plot(pi_vals, tra_para, stat='1', para='abtI', save=True, non_loc_path="plots_for_thesis/box3/bar_pi3.pdf", epe=epe)
    box.make_nn_plot(nn_vals, tra_para, stat='1', para='abtI', save=True, non_loc_path="plots_for_thesis/box3/box_nn3.pdf", epe=epe)


#second box plots
def fif_box():
    epe=80
    inf_vals, re_vals, pi_vals, nn_vals, tra_para = box.pros_data(stat='01', para='abtI', epe=epe, lr="0")
    bar.make_inf_plot(inf_vals, tra_para, stat='01', para='abtI', save=True, non_loc_path="plots_for_thesis/box4/bar_inf4.pdf", epe=epe)
    box.make_re_plot(re_vals, tra_para, stat='01', para='abtI', save=True, non_loc_path="plots_for_thesis/box4/box_re4.pdf", epe=epe)
    bar.make_pi_plot(pi_vals, tra_para, stat='01', para='abtI', save=True, non_loc_path="plots_for_thesis/box4/bar_pi4.pdf", epe=epe)
    box.make_nn_plot(nn_vals, tra_para, stat='01', para='abtI', save=True, non_loc_path="plots_for_thesis/box4/box_nn4.pdf", epe=epe)
    
    
#second box plots
def sixth_box():
    epe=80
    inf_vals, re_vals, pi_vals, nn_vals, tra_para = box.pros_data(stat='1', para='abtI', epe=epe)
    # box.make_inf_plot(inf_vals, tra_para, stat='1', para='abtI', save=True, non_loc_path="plots_for_thesis/box5/bar_inf5.pdf", epe=epe)
    bar.make_inf_plot(inf_vals, tra_para, stat='1', para='abtI', save=True, non_loc_path="plots_for_thesis/box5/bar_inf5.pdf", epe=epe)
    box.make_re_plot(re_vals, tra_para, stat='1', para='abtI', save=True, non_loc_path="plots_for_thesis/box5/box_re5.pdf", epe=epe)
    bar.make_pi_plot(pi_vals, tra_para, stat='1', para='abtI', save=True, non_loc_path="plots_for_thesis/box5/bar_pi5.pdf", epe=epe)
    box.make_pi_plot(pi_vals, tra_para, stat='1', para='abtI', save=True, non_loc_path="plots_for_thesis/box5/box_pi5.pdf", epe=epe)
    box.make_nn_plot(nn_vals, tra_para, stat='1', para='abtI', save=True, non_loc_path="plots_for_thesis/box5/box_nn5.pdf", epe=epe)
    
def undef():
    # epe=40
    path = Path("fhn_res_clus/fhn_res_s-0_v-abtI_n0_e40/expe_0")
    # inf_vals, re_vals, pi_vals, nn_vals, tra_para = box.pros_data(stat='1', para='abtI', epe=epe)
    make_comb_plot_v2(path, do_tit=False, non_loc_path="plots_for_thesis/undef.pdf")


def plot_fhn():
    t = np.linspace(0, 999, 1000)
    y0 = fitzhugh_nagumo_model(t)
    
    fig, ax = plt.subplots()
    ax.plot(t, y0[:,0], label='v')
    ax.plot(t, y0[:,1], label='w')
    ax.legend(loc="lower left")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Potential [mV] / Current [mA]")
    plt.savefig("plots_for_thesis/fhn_tar.pdf")
    plt.show()
    

def plot_phase():
    t = np.linspace(0, 999, 1000)
    y0 = fitzhugh_nagumo_model(t)
    y1 = fitzhugh_nagumo_model(t, a=-.27, b=1., tau=18, Iext=.25)
    y2 = fitzhugh_nagumo_model(t, a=-.35, b=1.3, tau=25, Iext=.3)
    y3 = fitzhugh_nagumo_model(t, a=-.27, b=1.3, tau=18, Iext=.3)
    # y4 = fitzhugh_nagumo_model(t, Iext=.3)
    # y2 = fitzhugh_nagumo_model(t, b=1.41)
    
    fig, axs = plt.subplots(1,4, figsize=(20,4))
    ax = axs.flatten()
    ax[0].plot(y0[:,0], y0[:,1], label=r"$a = -0.3,$ $b = 1.1,$"+"\n"+r"$\tau = 20,$ $I_{ext} = 0.23$")
    ax[1].plot(y1[:,0], y1[:,1], label=r"$a = -0.27,$ $b = 1.0,$"+"\n"+r"$\tau = 18,$ $I_{ext} = 0.25$")
    ax[2].plot(y2[:,0], y2[:,1], label=r"$a = -0.35,$ $b = 1.3,$"+"\n"+r"$\tau = 25,$ $I_{ext} = 0.3$")
    ax[3].plot(y3[:,0], y3[:,1], label=r"$a = -0.27,$ $b = 1.3,$"+"\n"+r"$\tau = 18,$ $I_{ext} = 0.3$")
    # ax.plot(y4[:,0], y4[:,1], label='five')
    
    tits = [r"$a = -0.3,$ $b = 1.1,$"+"\n"+r"$\tau = 20,$ $I_{ext} = 0.23$", r"$a = -0.27,$ $b = 1.0,$"+"\n"+r"$\tau = 18,$ $I_{ext} = 0.25$",
            r"$a = -0.35,$ $b = 1.3,$"+"\n"+r"$\tau = 25,$ $I_{ext} = 0.3$", r"$a = -0.27,$ $b = 1.3,$"+"\n"+r"$\tau = 18,$ $I_{ext} = 0.3$"]

    ax[0].set_ylabel(r"$w$")    
    for i in range(4):
        # ax[i].legend(prop={'size': 20})
        ax[i].set_xlabel(r"$v$")
        ax[i].set_title(tits[i])
        for item in ([ax[i].title, ax[i].xaxis.label, ax[i].yaxis.label] +
                     ax[i].get_xticklabels() + ax[i].get_yticklabels()):
            item.set_fontsize(20)
        
    plt.savefig("plots_for_thesis/fhn_phase.pdf", bbox_inches='tight')
    plt.show()
    
def plot_phase2():
    t = np.linspace(0, 999, num=1000)
    y0 = fitzhugh_nagumo_model(t)
    y1 = fitzhugh_nagumo_model(t, a=-.27, b=1., tau=18, Iext=.25)
    y2 = fitzhugh_nagumo_model(t, a=-.35, b=1.3, tau=25, Iext=.3)
    y3 = fitzhugh_nagumo_model(t, a=-.27, b=1.3, tau=18, Iext=.3)


    fig, axs = plt.subplots(1,4, figsize=(19,4), sharex=True, sharey=True)
    ax = axs.flatten()
    ax[0].plot(y0[:,0], y0[:,1], label=r"$a = -0.3,$ $b = 1.1,$"+"\n"+r"$\tau = 20,$ $I_{ext} = 0.23$")
    ax[1].plot(y1[:,0], y1[:,1], label=r"$a = -0.27,$ $b = 1.0,$"+"\n"+r"$\tau = 18,$ $I_{ext} = 0.25$")
    ax[2].plot(y2[:,0], y2[:,1], label=r"$a = -0.35,$ $b = 1.3,$"+"\n"+r"$\tau = 25,$ $I_{ext} = 0.3$")
    ax[3].plot(y3[:,0], y3[:,1], label=r"$a = -0.27,$ $b = 1.3,$"+"\n"+r"$\tau = 18,$ $I_{ext} = 0.3$")
    
    fig.subplots_adjust(wspace=.02)
    
    tits = [r"$a = -0.3,$ $b = 1.1,$"+"\n"+r"$\tau = 20,$ $I_{ext} = 0.23$", r"$a = -0.27,$ $b = 1.0,$"+"\n"+r"$\tau = 18,$ $I_{ext} = 0.25$",
            r"$a = -0.35,$ $b = 1.3,$"+"\n"+r"$\tau = 25,$ $I_{ext} = 0.3$", r"$a = -0.27,$ $b = 1.3,$"+"\n"+r"$\tau = 18,$ $I_{ext} = 0.3$"]

    ax[0].set_ylabel(r"$w$")
    
    for i in range(4):
        # ax[i].legend(prop={'size': 20})
        ax[i].set_xlabel(r"$v$")
        ax[i].set_title(tits[i])
        # ax[i].set_xlim([xmin*1.2, xmax*1.2])
        # ax[i].set_ylim([ymin*1.1, ymax*1.1])
        # if i != 0:
        #     ax[i].yaxis.set_visible(False)
        
        for item in ([ax[i].title, ax[i].xaxis.label, ax[i].yaxis.label] +
                     ax[i].get_xticklabels() + ax[i].get_yticklabels()):
            item.set_fontsize(20)
    
    plt.savefig("plots_for_thesis/fhn_phase.pdf", bbox_inches='tight')
    plt.show()
    
    
    fig, axs = plt.subplots(1,4, figsize=(19,4), sharex=True, sharey=True)
    ax = axs.flatten()
    ax[0].plot(t, y0[:,0], y0[:,1])
    ax[1].plot(t, y1[:,0], y1[:,1])
    ax[2].plot(t, y2[:,0], y2[:,1])
    ax[3].plot(t, y3[:,0], y3[:,1])
    
    fig.subplots_adjust(wspace=.02)
    
    ax[0].set_ylabel(r"$v, w$")
    ax[3].legend(["v", "w"], prop={'size': 20})
    
    for i in range(4):
        # ax[i].legend(prop={'size': 20})
        ax[i].set_xlabel(r"$t$")
        # ax[i].set_ylim([xmin*1.1, xmax*1.1])
        # ax[i].set_title(tits[i])
        for item in ([ax[i].title, ax[i].xaxis.label, ax[i].yaxis.label] +
                     ax[i].get_xticklabels() + ax[i].get_yticklabels()):
            item.set_fontsize(20)
    
    plt.savefig("plots_for_thesis/fhn_time.pdf", bbox_inches='tight')
    plt.show()

def plot_phase0():
    t = np.linspace(0, 999, 1000)
    y0 = fitzhugh_nagumo_model(t)
    y1 = fitzhugh_nagumo_model(t, a=-.27, b=1., tau=18, Iext=.25)
    y2 = fitzhugh_nagumo_model(t, a=-.35, b=1.3, tau=25, Iext=.3)
    # y3 = fitzhugh_nagumo_model(t, a=-.27, b=1.3, tau=18, Iext=.3)
    # y4 = fitzhugh_nagumo_model(t, Iext=.3)
    # y2 = fitzhugh_nagumo_model(t, b=1.41)
    
    fig, axs = plt.subplots(1,3, figsize=(16,4))
    ax = axs.flatten()
    ax[0].plot(y0[:,0], y0[:,1], label=r"$a = -0.3,$ $b = 1.1,$"+"\n"+r"$\tau = 20,$ $I_{ext} = 0.23$")
    ax[1].plot(y1[:,0], y1[:,1], label=r"$a = -0.27,$ $b = 1.0,$"+"\n"+r"$\tau = 18,$ $I_{ext} = 0.25$")
    ax[2].plot(y2[:,0], y2[:,1], label=r"$a = -0.35,$ $b = 1.3,$"+"\n"+r"$\tau = 25,$ $I_{ext} = 0.3$")
    # ax[3].plot(y3[:,0], y3[:,1], label=r"$a = -0.27,$ $b = 1.3,$"+"\n"+r"$\tau = 18,$ $I_{ext} = 0.3$")
    # ax.plot(y4[:,0], y4[:,1], label='five')
    
    tits = [r"$a = -0.3,$ $b = 1.1,$"+"\n"+r"$\tau = 20,$ $I_{ext} = 0.23$", r"$a = -0.27,$ $b = 1.0,$"+"\n"+r"$\tau = 18,$ $I_{ext} = 0.25$",
            r"$a = -0.35,$ $b = 1.3,$"+"\n"+r"$\tau = 25,$ $I_{ext} = 0.3$"] 
    
    ax[0].set_ylabel(r"$w$")
    for i in range(3):
        # ax[i].legend()
        ax[i].set_xlabel(r"$v$")
        
        ax[i].set_title(tits[i])
        for item in ([ax[i].title, ax[i].xaxis.label, ax[i].yaxis.label] +
                     ax[i].get_xticklabels() + ax[i].get_yticklabels()):
            item.set_fontsize(20)
    plt.savefig("plots_for_thesis/fhn_phase0.pdf", bbox_inches='tight')
    plt.show()
    
    # fig, ax = plt.subplots()
    # ax.plot(y0[:,0], y0[:,1], label='one')
    # ax.plot(y1[:,0], y1[:,1], label='two')
    # ax.plot(y2[:,0], y2[:,1], label='three')
    # ax.plot(y3[:,0], y3[:,1], label='four')
    # ax.plot(y4[:,0], y4[:,1], label='five')
    # ax.legend()
    # ax.set_xlabel("v")
    # ax.set_ylabel("w")
    # # plt.savefig("plots_for_thesis/ft_comp.pdf")
    # plt.show()
    

#figure for sec:ft
def plot_features():

    t = np.linspace(0, 999, 1000)
    y = fitzhugh_nagumo_model(t)
    
    fig, ax = plt.subplots()
    ax.plot(t, y[:,0], label='v')
    ax.plot(t, np.sin(0.0173*2*np.pi*t), '--', label='k=17.3')
    # ax.plot(t, y[:,1], label='w')
    ax.legend()
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Value")
    plt.savefig("plots_for_thesis/ft_comp.pdf")
    plt.show()




def plot_comp():
    
    t = np.linspace(0, 999, 1000)
    y0 = fitzhugh_nagumo_model(t, b=1.4)
    y1 = fitzhugh_nagumo_model(t, b=1.4051)
    
    fig, ax = plt.subplots()
    ax.plot(t, y0[:,0], label='b=1.4')
    ax.plot(t, y1[:,0], label='b=1.4051')
    
    ax.legend()
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Potential (mV)")
    plt.savefig("plots_for_thesis/inst_comp.pdf")
    plt.show()
    
# first_good()
bad_figs()

# first_box()
# sec_box()
# third_box()
# four_box()
# fif_box()
# sixth_box()

# out_dom("plots_for_thesis")

# undef()
# plot_phase()    
# plot_phase0()
# plot_phase2()
# plot_fhn()

# plot_features()
# plot_comp()

    
    