# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:13:48 2021

@author: benda
"""
from pathlib import Path
import numpy as np
from scipy.integrate import odeint
import deepxde as dde
from deepxde.backend import tf
import os
import time
import shutil
import matplotlib.pyplot as plt
import json
import sys

from postprocessing import saveplot
from make_plots import make_plots
from make_one_plot import make_one_plot, plot_losses

np.random.seed(2)


def fitzhugh_nagumo_model(
    t, a=-0.3, b=1.1, tau=20, Iext=0.23, x0 = [0, 0]   # maybe try different init values
):
    def func(x, t):
        #shouldn't v^3 be divided by 3?
        return np.array([x[0] - x[0] ** 3 / 3 - x[1] + Iext, (x[0] - a - b * x[1]) / tau])

    return odeint(func, x0, t)


def create_observations(data_t, data_y, geom):

    n = len(data_t)
    # Create a random array of size n/4 selecting indices between 1 and n-1
    idx = np.random.choice(np.arange(1, n - 1), size=n // 4, replace=False)
    # Add the last point to the list
    idx = np.append(idx, [0, n - 1])

    # np.savetxt(
    #     os.path.join(savename, "input.dat"),
    #     np.hstack((data_t[idx], data_y[idx, 0:1], data_y[idx, 1:2])),
    # )

    # Turn these timepoints into a set of points
    ptset = dde.bc.PointSet(data_t[idx])
    # Create a function that returns true when a point is part of the point set
    inside = lambda x, _: ptset.inside(x)

    # Create the observations by using the point set
    observe_y4 = dde.DirichletBC(
        geom, ptset.values_to_func(data_y[idx, 0:1]), inside, component=0
    )
    observe_y5 = dde.DirichletBC(
        geom, ptset.values_to_func(data_y[idx, 1:2]), inside, component=1
    )

    return observe_y4, observe_y5


def create_data(data_t, data_y, var_trainable=[True, True, False, False], var_modifier=[-.25, 1.1, 20, 0.23]):
    
    # Define the variables and constants in the model
    var_list = [] # a, b, tau, Iext
    #we want to include the possibility for the variables to be both trainable and constant
    for i in range(len(var_trainable)):
        if var_trainable[i]:
            #try having a and b be tanh()
            var = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) * var_modifier[i]
        else:
            var = tf.Variable(var_modifier[i], trainable=False, dtype=tf.float32)
        var_list.append(var)
        
    """
    a = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) * - 0.25
    b = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) 
    # tau = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) * 10
    # Iext = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) * 0.18
    tau = tf.Variable(20, trainable=False, dtype=tf.float32)
    Iext = tf.Variable(0.23, trainable=False, dtype=tf.float32)
    var_list = [a, b, tau, Iext]
    """

    def ODE(t, y):
        v1 = y[:, 0:1] - y[:, 0:1] ** 3 / 3 - y[:, 1:2] + var_list[3]
        v2 = (y[:, 0:1] - var_list[0] - var_list[1] * y[:, 1:2]) / var_list[2]
        return [
            tf.gradients(y[:, 0:1], t)[0] - v1,
            tf.gradients(y[:, 1:2], t)[0] - v2,
        ]

    # Create a time domain from first to last timepoint
    geom = dde.geometry.TimeDomain(data_t[0, 0], data_t[-1, 0])

    # Set the points on the right boundary
    # We assume these points are known
    def boundary(x, _):
        return np.isclose(x[0], data_t[-1, 0])

    y1 = data_y[-1]
    # Question: Does it matter which point we choose?
    bc0 = dde.DirichletBC(geom, lambda X: y1[0], boundary, component=0)
    bc1 = dde.DirichletBC(geom, lambda X: y1[1], boundary, component=1)

    observe_y4, observe_y5 = create_observations(data_t, data_y, geom)
    
    data = dde.data.PDE(  
        geom,
        ODE,
        [bc0, bc1, observe_y4, observe_y5],  # list of boundary conditions
        anchors=data_t,
    )
    
    
    # print("Data: ", len(data.test()), len(data.test()[0]), len(data.test()[0]), len(data.test()[0]))
    # print(data.test())
    return data, var_list


def create_nn(data_y, k_vals=[0.0173], nn_layers=3, nn_nodes=128, do_output_transform = True, 
              do_t_input_transform = True):
    
    # Feed-forward neural networks
    net = dde.maps.FNN(
        # layer_size=[1, 128, 128, 128, 2],
        layer_size=[1] + [nn_nodes]*nn_layers + [2],
        activation="swish",
        kernel_initializer="Glorot normal",
        # kernel_initializer="He normal",
    )
    
    #try to visualize the output with and without feature_transform
    
    def feature_transform(t):
        features = [] # np.zeros(len(k_vals) + 1)
        if do_t_input_transform:    
            # print("here")
            features.append(t) #[0] = t
            
        for k in range(len(k_vals)):
            features.append( tf.sin(k_vals[k] * 2*np.pi*t) )
        return tf.concat(features, axis=1)
    """
            (
                # t,
                # tf.sin(0.01 * t),
                # tf.sin(0.05 * t),
                # tf.sin(0.1 * t),
                # tf.sin(0.15 * t),
                # tf.sin(0.005*2*np.pi*t),
                # tf.sin(0.01*2*np.pi*t),
                tf.sin(0.013*2*np.pi*t),
                # tf.sin(0.02*2*np.pi*t),
                # tf.sin(k * t),
                # tf.sin(0.05*2*np.pi*t),
                #try f.exs. tf.sin(0.15 * t + 5),
            ),
            axis=1,
        )
    """

    net.apply_feature_transform(feature_transform)

    def output_transform(t, y):
        # Weights in the output layer are chosen as the magnitudes
        # of the mean values of the ODE solution
        return data_y[0] + tf.math.tanh(t/100) * tf.constant([0.1, 0.1]) * y
    
    if do_output_transform:    
        net.apply_output_transform(output_transform)
        
    return net


def create_callbacks(var_list, savename):
    # Save model after 1000 ephocs
    checkpointer = dde.callbacks.ModelCheckpoint(
        os.path.join(savename, "model/model.ckpt"),
        verbose=1,
        save_better_only=True,
        period=1000,
    )
    # Save variables after 1000 epochs
    variable = dde.callbacks.VariableValue(
        var_list,
        period=1000,
        filename=os.path.join(savename, "variables.dat"),
        precision=3,
    )
    return [checkpointer, variable]


def default_weights(noise, init_weights = [[1, 1], [1, 1], [1, 1]]):
    #init_weights are the wheights before noise is considered
    bc_weights = init_weights[1] # [1, 1]
    if noise >= 0.1:
        bc_weights = [w * 10 for w in bc_weights]

    data_weights = init_weights[2] # [1, 1]  
    # Large noise requires small data_weights
    if noise >= 0.1:
        data_weights = [w / 10 for w in data_weights]

    ode_weights = init_weights[0] # [1, 1] 
    # Large noise requires large ode_weights
    if noise > 0:
        ode_weights = [10 * w for w in ode_weights]

    return dict(
        bc_weights=bc_weights, data_weights=data_weights, ode_weights=ode_weights
    )


def train_model(model, weights, callbacks, first_num_epochs, sec_num_epochs, model_restore_path=None, lr=1e-3,
                batch_size=10):

    # First compile the model with ode weights set to zero
    model.compile(
        "adam",
        lr=lr,
        loss_weights=[0] * 2 + weights["bc_weights"] + weights["data_weights"],
    )
    # import IPython
    # IPython.embed()
    # And train
    model.train(epochs=int(first_num_epochs), display_every=int(first_num_epochs), batch_size=batch_size)
    
    # IPython.embed()
    # Now compile the model, but this time include the ode weights
    model.compile(
        "adam",
        lr=lr,
        loss_weights=weights["ode_weights"]
        + weights["bc_weights"]
        + weights["data_weights"],
    )

    losshistory, train_state = model.train(
        epochs=int(sec_num_epochs),
        display_every=100,
        callbacks=callbacks,
        disregard_previous_best=True,
        model_restore_path=model_restore_path,
        batch_size=batch_size,
    )
    return losshistory, train_state


def get_model_restore_path(restore, savename):
    #if you want to restore from a previous run
    if restore:
        #reads form the checkpoint text file
        with open(os.path.join(savename,"model/checkpoint"), 'r') as reader:
            inp = reader.read()
            restore_from = inp.split(" ")[1].split('"')[1]
        return os.path.join(savename,"model", restore_from)
    else:
        return None


def create_hyperparam_dict(
    savename,
    first_num_epochs,
    sec_num_epochs,
    var_trainable, 
    var_modifier,
    lr,
    init_weights,
    k_vals,
    do_output_transform,
    do_t_input_transform,
    batch_size,
):
    """
    This function creates a dictionary contianing all the hyperparameters, and 
    saves it to a file.
    """
    
    dictionary = dict(
        ode_weights=init_weights[0], bc_weights=init_weights[1], data_weights=init_weights[2],
        first_num_epochs=first_num_epochs, sec_num_epochs=sec_num_epochs,
        var_trainable=var_trainable, var_modifier=var_modifier, 
        k_vals=k_vals, lr=lr, do_output_transform=do_output_transform, do_t_input_transform=do_t_input_transform,
        batch_size=batch_size,
    )
    # np.savetxt(os.path.join(savename, "hyperparameters.dat"), dictionary)   
    with open(os.path.join(savename, "hyperparameters.dat"),'w') as data: 
        for key, value in dictionary.items(): 
            data.write('%s: %s\n' % (key, value))
    import pickle #try this later
    a_file = open("hyperparameters.pkl", "wb") 
    pickle.dump(dictionary, a_file)    
    a_file.close()


def pinn(
    data_t,
    data_y,
    noise,
    savename,
    restore=False,
    first_num_epochs=int(1e3),
    sec_num_epochs=int(1e5),
    var_trainable=[True, True, False, False], 
    var_modifier=[-.25, 1.1, 20, 0.23],
    lr=1e-3,
    init_weights = [[1, 1], [1, 1], [1, 1]],
    k_vals=[0.013],
    do_output_transform = False,
    do_t_input_transform = True,
    batch_size = 10,
):
   
    data, var_list = create_data(data_t, data_y, var_trainable, var_modifier)

    net = create_nn(data_y, k_vals, do_output_transform=do_output_transform, do_t_input_transform=do_t_input_transform)
    model = dde.Model(data, net)
    
    # import IPython
    # IPython.embed()
    
    #try plotting model.predict(t) 

    callbacks = create_callbacks(var_list, savename)

    weights = default_weights(noise, init_weights)
    model_restore_path = get_model_restore_path(restore, savename)
    
    create_hyperparam_dict(savename, first_num_epochs, sec_num_epochs, var_trainable, 
                           var_modifier, lr, init_weights, k_vals, do_output_transform,
                           do_t_input_transform, batch_size)
    
    losshistory, train_state = train_model(
        model, weights, callbacks, first_num_epochs, sec_num_epochs, model_restore_path, lr=lr, 
        batch_size=batch_size
    )

    saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=savename)
    
    # model.save(os.path.join(savename, "nn_model.dat"))
    
    var_list = [model.sess.run(v) for v in var_list]
    
    nn_pred = model.predict(data_t) 
    np.savetxt(
        os.path.join(savename, "neural_net_pred_last.dat"), np.hstack((data_t, nn_pred))
    )
    model.restore(get_model_restore_path(True, savename))
    nn_pred = model.predict(data_t) 
    np.savetxt(
        os.path.join(savename, "neural_net_pred_best.dat"), np.hstack((data_t, nn_pred))
    )
    
    return var_list


def generate_data(savename, true_values, t_vars, noise=0.0):
    # Generate data to be used as observations
    t = np.linspace(*t_vars)[:, None] 
    y = fitzhugh_nagumo_model(np.ravel(t), *true_values)
    np.savetxt(os.path.join(savename, "fitzhugh_nagumo.dat"), np.hstack((t, y)))
    # Add noise
    if noise > 0:
        std = noise * y.std(0)
        y[1:-1, :] += np.random.normal(0, std, (y.shape[0] - 2, y.shape[1]))
        np.savetxt(
            os.path.join(savename, "fitzhugh_nagumo_noise.dat"), np.hstack((t, y))
        )
    return t, y


def make_copy_of_program(savename):
    """
    From https://stackoverflow.com/questions/23321100/best-way-to-have-a-python-script-copy-itself/49210778    
    """
    
    # generate filename with timestring
    copied_script_name = time.strftime("%Y-%m-%d_%H%M") + '_' + os.path.basename(__file__)
    
    # copy script
    shutil.copy(__file__, os.path.join( savename, copied_script_name) )


def main():
    start = time.time()
    noise = 0.0
    # tf.device("gpu")
    savename = Path("fhn_res/fitzhugh_nagumo_res_bas10_2_90")
    # Create directory if not exist
    savename.mkdir(exist_ok=True)
    
    make_copy_of_program(savename)


    # a, b, tau, Iext
    a = -0.3
    b = 1.1
    tau = 20
    Iext = 0.23
    true_values = [a, b, tau, Iext]
    t_vars = [0, 999, 1000]

    t, y = generate_data(savename, true_values, t_vars, noise)
    
    # from IPython import embed
    # embed()


    # Train
    var_list = pinn(
        t,
        y,
        noise,
        savename,
        restore=False,
        first_num_epochs=10,
        sec_num_epochs=int(5e3),
        var_trainable=[False, True, False, False], #a, b, tau, Iext
        var_modifier=[-.3, .8, 20, 0.23], #a, b, tau, Iext
        # init_weights = [[1, 1], [0, 0], [0, 0]], # [[ode], [bc], [data]]
        init_weights = [[3, 60], [.01, .01], [1, 4.5]], # [[ode], [bc], [data]]
        k_vals=[0.0173], # tf.sin(k * 2*np.pi*t),
        # lr = 5e4,
        do_output_transform = True,
        do_t_input_transform = False,
        batch_size = 20,
        lr=1e-2,
    )

    # Prediction
    pred_y = fitzhugh_nagumo_model(np.ravel(t), *var_list)
    np.savetxt(
        os.path.join(savename, "fitzhugh_nagumo_pred.dat"), np.hstack((t, pred_y))
    )
    
    print("\n\nTotal runtime: {}".format(time.time() - start))

    print("Original values: ")
    print(f"(a, b tau, Iext) = {true_values}")

    print("Predicted values: ")
    print(f"(a, b tau, Iext) = {var_list}\n")

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title("$v$")
    ax[0].plot(t, y.T[0], label="True")
    ax[0].plot(t, pred_y.T[0], label="Predicted")

    ax[1].set_title("$w$")
    ax[1].plot(t, y.T[1], label="True")
    ax[1].plot(t, pred_y.T[1], label="Predicted")

    for axi in ax:
        axi.legend()
        axi.grid()

    fig.savefig(savename.joinpath("predicted_vs_true.pdf"))
    plt.show()    
    
    make_plots(savename)
    make_one_plot(savename)
    plot_losses(savename)
    


def plot_features():

    t = np.linspace(0, 999, 1000)
    y = fitzhugh_nagumo_model(t)
    fig, ax = plt.subplots()
    ax.plot(t, y[:,0])
    # ax.plot(t, np.sin(0.01 * t))
    # ax.plot(t, np.sin(0.05 * t))
    # ax.plot(t, np.sin(0.1 * t))
    ax.plot(t, np.sin(0.0173*2*np.pi*t))
    # ax.plot(t, np.sin(0.015*2*np.pi*t))
    # ax.plot(t, np.sin(0.012*2*np.pi*t))
    

    plt.show()


if __name__ == "__main__":
    main()
    # plot_features()
