# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 19:59:49 2022

@author: benda
"""

import numpy as np
import os
from pathlib import Path
import pickle
# import julia 
# julia.install()
from julia import Main

if __name__ == "__main__":
    # make_plots(Path("fhn_res/fhn_res_clus/fhn_res_s-01_v-abtI_n1_e50"), params=[0,1,2,3])
    Main.include("practical_identifiability_fhn.jl")
    
    for di in os.listdir('./fhn_res_clus'):
        if len(os.listdir('./fhn_res_clus/{}'.format(di))) < 10:
            print(di, len(os.listdir('./fhn_res_clus/{}'.format(di))))
        # if (di != "fhn_res_s-01_v-abtI_n10_e40") and (di != "fhn_res_s-01_v-ab_n0_e40"):
        for exe in os.listdir('./fhn_res_clus/{}'.format(di)):
            
            print(di, exe)
            
            path = Path("fhn_res_clus/{}/{}".format(di, exe) )
            with open(os.path.join(path, "hyperparameters.pkl"), "rb") as a_file:
                hdata = pickle.load(a_file)
            with open(os.path.join(path, "evaluation.pkl"), "rb") as a_file:
                edata = pickle.load(a_file)
            states = [i+1 for i in hdata['observed_states']]
            params = hdata['var_trainable']
            noise  = hdata['noise']
            found  = edata['found_param']
            
            try:    
                lowerbound, lab, err = Main.practical_identifiability_fhn(found, params, states, noise)
                dic = {'lowerbound':lowerbound, 'lab':lab, 'err':err}
                
                a_file = open(os.path.join(path, "pra_ident.pkl"), "wb") 
                pickle.dump(dic, a_file)    
                a_file.close()
                
                with open(os.path.join(path, "pra_ident.dat"),'w') as data: 
                    for key, value in dic.items(): 
                        data.write('%s: %s\n' % (key, value))
                
                
                out = "{}, ".format(','.join(str(i) for i in np.array(['k','v','w'])[states]))
                out += "{}:".format(','.join(str(i) for i in np.array(["a","b","τ","I_{ext}"])[np.where(params)[0]] ))
                
                for l in range(len(err)):    
                    out += " {} = {:.5e} ± {:.5e},".format(np.array(["a","b","τ","I_{ext}"])[np.where(params)[0]][l], lowerbound[l], err[l])
                out = out[:-1] + '.\n'
                print(out)
            except RuntimeError:
                print("Very bar fit.\n")
                
                out = np.array([np.inf,np.inf,np.inf,np.inf])[params]
                lowerbound, lab, err = (out,out,out)
                dic = {'lowerbound':lowerbound, 'lab':lab, 'err':err}
                
                a_file = open(os.path.join(path, "pra_ident.pkl"), "wb") 
                pickle.dump(dic, a_file)    
                a_file.close()
                
                with open(os.path.join(path, "pra_ident.dat"),'w') as data: 
                    for key, value in dic.items(): 
                        data.write('%s: %s\n' % (key, value))
        
    # make_plots(Path("fhn_res/fitzhugh_nagumo_res_all_01"), if_noise=True, params=[0,1,2,3])

    
    # print(Main.practical_identifiability_fhn([1.104, 19.98], [False,True,True,False], [1], 0.1))


