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

import beeler_reuter.beeler_reuter_1977_version06 as br_model 

from postprocessing import saveplot
from make_plots import make_plots
from make_one_plot import make_one_plot, plot_losses

np.random.seed(2)


def beeler_reuter_model(
    t, params, x0 = br_model.init_state_values()
):
    """
    Solves the Beeler Reuter ODE for the given time input and model parameters.
    
    Parameters
    ----------
    t : float
        Time input. Can also ba an array of floats.
    params : numpy array (?)
        The model parameters.
    x0 : list, optional
        The initial values. The default is br_model.init_state_values().

    Returns
    -------
    numpy array
        Solution.
        
    """
    
    def func(x, t):
        return br_model.rhs(t, x, params) # np.array([x[0] - x[0] ** 3 - x[1] + Iext, (x[0] - a - b * x[1]) / tau])
    return odeint(func, x0, t)


def create_observations(data_t, data_y, geom):
    """
    Generates synthetic data using observations objects. This represents the specific 
    timepoints where observations/measurements were made.

    Parameters
    ----------
    data_t : numpy array
        Potential timepoints.
    data_y : numpy array
        Measurements at those timepoints.
    geom : deepxde geometry.TimeDomain
        The geometry of the system.

    Returns
    -------
    observe_y0 : deepxde DirichletBC
        Measurements fot the first state.
    observe_y1 : deepxde DirichletBC
        Measurements fot the second state.

    """

    n = len(data_t)
    # Create a random array of size n/4 selecting indices between 1 and n-1
    idx = np.random.choice(np.arange(1, n - 1), size=n // 4, replace=False)
    # Add the last point to the list
    idx = np.append(idx, [0, n - 1])

    # Turn these timepoints into a set of points
    ptset = dde.bc.PointSet(data_t[idx])
    # Create a function that returns true when a point is part of the point set
    inside = lambda x, _: ptset.inside(x)

    # Create the observations by using the point set
    observes = []
    for i in range(len(data_y[0])):
        obs = dde.DirichletBC(
            geom, ptset.values_to_func(data_y[idx, i:i+1]), inside, component=i
        )
        observes.append(obs)
    
    return observes


def create_data(data_t, data_y, var_trainable, var_modifier, 
                scale_func = tf.math.softplus):
    """
    Function that generates all the required data, and sets up all data objects.

    Parameters
    ----------
    data_t : numpy array
        Potential timepoints.
    data_y : numpy array
        Measurements at those timepoints.
    var_trainable : list, optional
        List of which ODE-parameters should or shouldn't be trainable. The default is [True, True, False, False].
    var_modifier : list, optional
        List of modifiers for the ODE-parameters. The default is [-.25, 1.1, 20, 0.23].
    scale_func : function, optional
        Function for scailing the trainable parameters. The default is tf.math.softplus.

    Returns
    -------
    deepxde data.PDE, list
        First is a data object to use for all the training.
        Second is a list of tensorflow variables representing the ODE-params.

    """
    
    # Define the variables and constants in the model
    var_list = [] 
    #we want to include the possibility for the variables to be both trainable and constant
    for i in range(len(var_trainable)):
        if var_trainable[i]:
            var = scale_func(tf.Variable(0, trainable=True, dtype=tf.float32)) * var_modifier[i]
        else:
            var = tf.Variable(var_modifier[i], trainable=False, dtype=tf.float32)
        var_list.append(var)
    
    #the ode in tensorflow syntax
    def ODE(t, y):
         
        values = [] #np.zeros((8,), dtype=np.float_)        
        (
            E_Na,
            g_Na,
            g_Nac,
            g_s,
            IstimAmplitude,
            IstimEnd,
            IstimPeriod,
            IstimPulseDuration,
            IstimStart,
            C,
        ) = var_list
    
        # Expressions for the Sodium current component
        i_Na = (g_Nac + g_Na * (y[:, 0:1] * y[:, 0:1] * y[:, 0:1]) * y[:, 1:2] * y[:, 2:3]) * (-E_Na + y[:, 7:8])
    
        # Expressions for the m gate component
        alpha_m = (-47 - y[:, 7:8]) / (-1 + 0.009095277101695816 * tf.exp(-0.1 * y[:, 7:8]))
        beta_m = 0.7095526727489909 * tf.exp(-0.056 * y[:, 7:8])
        values.append( (1 - y[:, 0:1]) * alpha_m - beta_m * y[:, 0:1] )
    
        # Expressions for the h gate component
        alpha_h = 5.497962438709065e-10 * tf.exp(-0.25 * y[:, 7:8])
        beta_h = 1.7 / (1 + 0.1580253208896478 * tf.exp(-0.082 * y[:, 7:8]))
        values.append( (1 - y[:, 1:2]) * alpha_h - beta_h * y[:, 1:2] )
    
        # Expressions for the j gate component
        alpha_j = (
            1.8690473007222892e-10
            * tf.exp(-0.25 * y[:, 7:8])
            / (1 + 1.6788275299956603e-07 * tf.exp(-0.2 * y[:, 7:8]))
        )
        beta_j = 0.3 / (1 + 0.040762203978366204 * tf.exp(-0.1 * y[:, 7:8]))
        values.append( (1 - y[:, 2:3]) * alpha_j - beta_j * y[:, 2:3] )
    
        # Expressions for the Slow inward current component
        E_s = -82.3 - 13.0287 * tf.math.log(0.001 * y[:, 3:4])
        i_s = g_s * (-E_s + y[:, 7:8]) * y[:, 4:5] * y[:, 5:6]
        values.append( 7.000000000000001e-06 - 0.07 * y[:, 3:4] - 0.01 * i_s )
    
        # Expressions for the d gate component
        alpha_d = (
            0.095
            * tf.exp(1 / 20 - y[:, 7:8] / 100)
            / (1 + 1.4332881385696572 * tf.exp(-0.07199424046076314 * y[:, 7:8]))
        )
        beta_d = 0.07 * tf.exp(-44 / 59 - y[:, 7:8] / 59) / (1 + tf.exp(11 / 5 + y[:, 7:8] / 20))
        values.append( (1 - y[:, 4:5]) * alpha_d - beta_d * y[:, 4:5] )
    
        # Expressions for the f gate component
        alpha_f = (
            0.012
            * tf.exp(-28 / 125 - y[:, 7:8] / 125)
            / (1 + 66.5465065250986 * tf.exp(0.14992503748125938 * y[:, 7:8]))
        )
        beta_f = 0.0065 * tf.exp(-3 / 5 - y[:, 7:8] / 50) / (1 + tf.exp(-6 - y[:, 7:8] / 5))
        values.append( (1 - y[:, 5:6]) * alpha_f - beta_f * y[:, 5:6] )
    
        # Expressions for the Time dependent outward current component
        i_x1 = (
            0.0019727757115328517
            * (-1 + 21.75840239619708 * tf.exp(0.04 * y[:, 7:8]))
            * tf.exp(-0.04 * y[:, 7:8])
            * y[:, 6:7]
        )
    
        # Expressions for the X1 gate component
        alpha_x1 = (
            0.031158410986342627
            * tf.exp(0.08264462809917356 * y[:, 7:8])
            / (1 + 17.41170806332765 * tf.exp(0.05714285714285714 * y[:, 7:8]))
        )
        beta_x1 = (
            0.0003916464405623223
            * tf.exp(-0.05998800239952009 * y[:, 7:8])
            / (1 + tf.exp(-4 / 5 - y[:, 7:8] / 25))
        )
        values.append( (1 - y[:, 6:7]) * alpha_x1 - beta_x1 * y[:, 6:7] )
    
        # Expressions for the Time independent outward current component
        i_K1 = 0.0035 * (4.6000000000000005 + 0.2 * y[:, 7:8]) / (
            1 - 0.39851904108451414 * tf.exp(-0.04 * y[:, 7:8])
        ) + 0.0035 * (-4 + 119.85640018958804 * tf.exp(0.04 * y[:, 7:8])) / (
            8.331137487687693 * tf.exp(0.04 * y[:, 7:8]) + 69.4078518387552 * tf.exp(0.08 * y[:, 7:8])
        )
    
        # Expressions for the Stimulus protocol component
        print( type(t <= IstimEnd), t <= IstimEnd)
        Istim = (
            IstimAmplitude
            if t - IstimStart - IstimPeriod * tf.math.floor((t - IstimStart) / IstimPeriod)
            <= IstimPulseDuration
            and t <= IstimEnd
            and t >= IstimStart
            else 0
        )
        # IstimAmplitude=0.5, IstimEnd=50000.0, IstimPeriod=1000.0, 
        # IstimPulseDuration=1.0, IstimStart=10.0
    
        # Expressions for the Membrane component
        values.append( (-i_K1 - i_Na - i_s - i_x1 + Istim) / C )
        res = []
        for i in range(len(values)):
            res.append(tf.gradients(y[:, i:i+1], t)[0] - values[i])
    
        # Return results
        return res
    

    # Create a time domain from first to last timepoint
    geom = dde.geometry.TimeDomain(data_t[0, 0], data_t[-1, 0])

    # Set the points on the right boundary
    # We assume these points are known
    def boundary(x, _):
        return np.isclose(x[0], data_t[-1, 0])
    
    y1 = data_y[-1]
    # Question: Does it matter which point we choose?
    bcs = []
    for i in range(len(y1)):
        bcs.append(dde.DirichletBC(geom, lambda X: y1[i], boundary, component=i))

    observes = create_observations(data_t, data_y, geom)
    
    data = dde.data.PDE(  
        geom,
        ODE,
        [*bcs, *observes],  # list of boundary conditions
        anchors=data_t,
    )

    return data, var_list


def create_nn(data_y, nn_layers=3, nn_nodes=128, activation = "swish", kernel_initializer="He normal", 
              do_t_input_transform = True, k_vals=[0.0173], do_output_transform = True):
    """
    Creates a neural network object. 

    Parameters
    ----------
    data_y : numpy array
        Known measurements.
    nn_layers : Int, optional
        Number of layers. The default is 3.
    nn_nodes : Int, optional
        Number of nodes in each layer. The default is 128.
    activation : string, optional
        The activation function. The default is "swish".
    kernel_initializer : string, optional
        The initialization of the nn-parmaeters. The default is "He normal".
    do_t_input_transform : Bool, optional
        Wheter we should use t in input transformation. The default is True.
    k_vals : list, optional
        Values for the input transformation. The default is [0.0173].
    do_output_transform : Bool, optional
        Wheter we should use the output transformation. The default is True.

    Returns
    -------
    deepxde maps.FNN
        An object for a feed forward neural network.
    """
    
    # Feed-forward neural networks
    net = dde.maps.FNN(
        layer_size=[1] + [nn_nodes]*nn_layers + [8],
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    
    def feature_transform(t):
        """
        Helper function for the feature transformation.
        """
        features = [] 
        if do_t_input_transform: #if we want to include unscaled as well
            features.append(t) #[0] = t
            
        for k in range(len(k_vals)):
            features.append( tf.sin(k_vals[k] * 2*np.pi*t) )
        return tf.concat(features, axis=1)
   

    net.apply_feature_transform(feature_transform)

    def output_transform(t, y):
        """
        Helper function for the output transformation.
        """
        # Weights in the output layer are chosen as the magnitudes
        # of the mean values of the ODE solution
        # TODO: this might need to change
        return data_y[0] + tf.math.tanh(t) * tf.constant([.1]*8) * y
        # return data_y[0] + tf.sin(k_vals[0] * 2*np.pi*t) * tf.constant([.1, .1]) * y
    
    if do_output_transform:    
        net.apply_output_transform(output_transform)
        
    return net


def create_callbacks(var_list, savename, save_every=100):
    """
    Helper function for saving the reults during the training. 
    """
    # Save model after 100 ephocs
    checkpointer = dde.callbacks.ModelCheckpoint(
        os.path.join(savename, "model/model.ckpt"),
        verbose=1,
        save_better_only=True,
        period=save_every,
    )
    # Save variables after 100 epochs
    variable = dde.callbacks.VariableValue(
        var_list,
        period=save_every,
        filename=os.path.join(savename, "variables.dat"),
        precision=3,
    )
    return [checkpointer, variable]


def default_weights(noise, init_weights = [[1, 1], [1, 1], [1, 1]]):
    """
    Makes the weights into a dictionary, and applies scailing if there's noise.

    Parameters
    ----------
    noise : float
        The amount of noise.
    init_weights : list, optional
        List of weights before noise is considered. The default is [[1, 1], [1, 1], [1, 1]].

    Returns
    -------
    dict
        Dictionary of the weights.

    """
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


def train_model(model, weights, callbacks, first_num_epochs = int(1e3), 
                sec_num_epochs = int(1e5), model_restore_path=None, lr=1e-3,
                batch_size=10, display_every=100):
    """
    Function that actually trains the PINN.

    Parameters
    ----------
    model : deepxde Model
        Model object.
    weights : dict
        Dictionary of weights.
    callbacks : list
        Callbacks.
    first_num_epochs : int, optional
        Number of epochs to train without ODE-loss. The default is int(1e3).
    sec_num_epochs : int, optional
        Number of epochs to train with ODE-loss. The default is int(1e5).
    model_restore_path : TYPE, optional
        DESCRIPTION. The default is None.
    lr : float, optional
        The learning rate. The default is 1e-3.
    batch_size : int, optional
        Batch size for minibatching. The default is 10.
    display_every : int, optional
        How often the result should be displayed/saved. The default is 100.

    Returns
    -------
    losshistory : TYPE
        DESCRIPTION.
    train_state : TYPE
        DESCRIPTION.

    """
    
    # First compile the model with ode weights set to zero
    model.compile(
        "adam",
        lr=lr,
        loss_weights=[0] * 8 + weights["bc_weights"] + weights["data_weights"],
    )
    # And train
    model.train(epochs=int(first_num_epochs), display_every=int(first_num_epochs), batch_size=batch_size)
    
    # Now compile the model, but this time include the ode weights
    model.compile(
        "adam",
        lr=lr,
        loss_weights=weights["ode_weights"]
        + weights["bc_weights"]
        + weights["data_weights"],
    )
    
    # And train
    losshistory, train_state = model.train(
        epochs=int(sec_num_epochs),
        display_every=display_every,
        callbacks=callbacks,
        disregard_previous_best=True,
        model_restore_path=model_restore_path,
        batch_size=batch_size,
    )
    return losshistory, train_state


def get_model_restore_path(restore, savename):
    """
    Helper function for getting the path for when restoring a previous run.
    """
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
    var_trainable=[False, True, True, True, False, False, False, False, False, False], 
    var_modifier=[50.0, 0.01, 1e-05, 0.0001, 0.5, 50000.0, 1000.0, 1.0, 10.0, 0.01], 
    lr=1e-2,
    ode_weights = [1, 1, 1, 1, 1, 1, 1, 1],
    bc_weights = [1]*8, # [1, 1, 1, 1, 1, 1, 1, 1],
    data_weights = [1, 1, 1, 1, 1, 1, 1, 1],
    k_vals=[0.013],
    do_output_transform = False,
    do_t_input_transform = True,
    batch_size = 10,
    nn_layers=3,
    nn_nodes=128,
    display_every=100,
):
    """
    Function for seting up and solving the PINN.
    TODO: update defaults.

    Parameters
    ----------
    data_t : numpy array
        Timepoints.
    data_y : numpy array
        Measurements at those timepoints.
    noise : float
        How much noise to add.
    savename : pathlib Path
        Path for saving information.
    restore : Bool, optional
        Wheter we should restore the NN from a previous run. The default is False.
    first_num_epochs : int, optional
        Number of epochs to train without ODE-loss. The default is int(1e3).
    sec_num_epochs : int, optional
        Number of epochs to train with ODE-loss. The default is int(1e5).
    var_trainable : list, optional
        List of which ODE-parameters should or shouldn't be trainable. The default is [True, True, False, False].
    var_modifier : list, optional
        List of modifiers for the ODE-parameters. The default is [-.25, 1.1, 20, 0.23].
    lr : float, optional
        The learning rate. The default is 1e-2.
    init_weights : list, optional
        List of weights before noise is considered. The default is [[1, 1], [1, 1], [1, 1]].
    do_t_input_transform : Bool, optional
        Wheter we should use t in input transformation. The default is True.
    k_vals : list, optional
        Values for the input transformation. The default is [0.0173].
    do_output_transform : Bool, optional
        Wheter we should use the output transformation. The default is True.
    batch_size : int, optional
        Batch size for minibatching. The default is 10.
    nn_layers : Int, optional
        Number of layers. The default is 3.
    nn_nodes : Int, optional
        Number of nodes in each layer. The default is 128.
    display_every : int, optional
        How often the result should be displayed/saved. The default is 100.

    Returns
    -------
    var_list : list
        Prediction of the ODE parameters.

    """
   
    init_weights = [ode_weights, bc_weights, data_weights]
   
    data, var_list = create_data(data_t, data_y, var_trainable, var_modifier)

    net = create_nn(data_y, k_vals=k_vals, do_output_transform=do_output_transform, 
                    do_t_input_transform=do_t_input_transform, nn_layers=nn_layers,
                    nn_nodes=nn_nodes)
    model = dde.Model(data, net)

    callbacks = create_callbacks(var_list, savename, display_every)

    weights = default_weights(noise, init_weights)
    model_restore_path = get_model_restore_path(restore, savename)
    
    create_hyperparam_dict(savename, first_num_epochs, sec_num_epochs, var_trainable, 
                           var_modifier, lr, init_weights, k_vals, do_output_transform,
                           do_t_input_transform, batch_size)
    
    losshistory, train_state = train_model(
        model, weights, callbacks, first_num_epochs, sec_num_epochs, model_restore_path, lr=lr, 
        batch_size=batch_size, display_every=display_every,
    )

    saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=savename)
        
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
    """
    Solves the Beeler-Reuter ODE for the given timepoints, and adds noise.

    Parameters
    ----------
    savename : pathlib Path
        Path for saving the input data.
    true_values : list
        The true values of the ODE.
    t_vars : list
        List of vaiables to create a numpy linspace for the time.
    noise : float, optional
        The amount of noise. The default is 0.0.

    Returns
    -------
    t : numpy array
        The timepoints.
    y : numpy array
        The solution at the given timepoints.

    """
    # Generate data to be used as observations
    t = np.linspace(*t_vars)[:, None] 
    y = beeler_reuter_model(np.ravel(t), true_values)
    np.savetxt(os.path.join(savename, "beeler_reuter.dat"), np.hstack((t, y)))
    
    # Add noise
    # todo: noise needs updating
    if noise > 0:
        std = noise * y.std(0)
        y[1:-1, :] += np.random.normal(0, std, (y.shape[0] - 2, y.shape[1]))
        np.savetxt(
            os.path.join(savename, "beeler_reuter_noise.dat"), np.hstack((t, y))
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
    """ 
    Main function.
    """
    
    start = time.time()
    noise = 0.0
    savename = Path("br_res/br_res_00")
    # Create directory if not exist
    savename.mkdir(exist_ok=True)
    
    make_copy_of_program(savename)


    # a, b, tau, Iext
    # a = -0.3
    # b = 1.1
    # tau = 20
    # Iext = 0.23
    true_values = br_model.init_parameter_values() # [a, b, tau, Iext]
    t_vars = [0, 999, 2000]

    t, y = generate_data(savename, true_values, t_vars, noise)
    
    # Train
    var_list = pinn(
        t,
        y,
        noise,
        savename,
        restore=False,
        first_num_epochs=100,
        sec_num_epochs=int(8e3),
        # E_Na, g_Na, g_Nac, g_s, IstimAmplitude, IstimEnd, IstimPeriod, 
        # IstimPulseDuration, IstimStart, C
        var_trainable=[False, True, True, True, False, False, False, False, False, False], 
        var_modifier=[50.0, 0.01, 1e-05, 0.0001, 0.5, 50000.0, 1000.0, 1.0, 10.0, 0.01], 
        ode_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        bc_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        data_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        k_vals=[1], #[0.0173], # tf.sin(k * 2*np.pi*t),
        do_output_transform = True,
        do_t_input_transform = False,
        batch_size = 50,
        lr=1e-3,
        nn_layers=3,
        nn_nodes=64,
    )

    # Prediction
    pred_y = beeler_reuter_model(np.ravel(t), var_list)
    np.savetxt(
        os.path.join(savename, "beeler_reuter_pred.dat"), np.hstack((t, pred_y))
    )
    
    runtime = time.time() - start
    print("\n\nTotal runtime: {}".format(runtime))

    print("Original values: ")
    print(f"(E_Na, g_Na, g_Nac, g_s, IstimAmplitude, IstimEnd, IstimPeriod, IstimPulseDuration, IstimStart, C) = {true_values}")

    print("Predicted values: ")
    print(f"(E_Na, g_Na, g_Nac, g_s, IstimAmplitude, IstimEnd, IstimPeriod, IstimPulseDuration, IstimStart, C) = {var_list}\n")

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
    y = beeler_reuter_model(t)
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
