# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:59:30 2022

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt
from make_one_plot import make_comb_plot, plot_losses
from pathlib import Path
from make_bar_plot import pros_data, make_inf_plot, make_re_plot
# from fitzhugh_nagumo import fitzhugh_nagumo
from fhn_pinn import fitzhugh_nagumo_model
from make_plots import make_plots


#first figure
path0 = Path("fhn_res_clus/fhn_res_s-01_v-a_n0_e40/expe_9")
make_comb_plot(path0, non_loc_path="plots_for_thesis/comp_plot0.pdf")
plot_losses(path0, non_loc_path="plots_for_thesis/loss_plot0.pdf")
make_plots(path0, non_loc_path=Path("plots_for_thesis/epoch_chang"), params=[0], do_re_change=True, do_param_change=True)



#second figure
# inf_vals, re_vals, pi_vals, tra_para = pros_data(stat='01', para='a')
# make_inf_plot(inf_vals, tra_para, stat='01', para='a', save=True, non_loc_path="plots_for_thesis/bar_inf0.pdf")
# make_re_plot(re_vals, tra_para, stat='01', para='a', save=True, non_loc_path="plots_for_thesis/bar_re0.pdf")


# #third figure
# inf_vals, re_vals, pi_vals, tra_para = pros_data(stat='01', para='abtI')
# make_inf_plot(inf_vals, tra_para, stat='01', para='abtI', save=True, non_loc_path="plots_for_thesis/bar_inf1.pdf")
# make_re_plot(re_vals, tra_para, stat='01', para='abtI', save=True, non_loc_path="plots_for_thesis/bar_re1.pdf")



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

plot_features()