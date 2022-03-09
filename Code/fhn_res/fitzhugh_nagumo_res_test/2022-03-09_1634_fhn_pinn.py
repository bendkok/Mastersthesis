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

from postprocessing import saveplot
from make_plots import make_plots
from make_one_plot import make_one_plot, plot_losses, make_samp_plot
from evaluate import evaluate

np.random.seed(2)


def fitzhugh_nagumo_model(
    t, a=-0.3, b=1.1, tau=20, Iext=0.23, x0 = [0, 0]   
):
    """
    Solves the Fitzhugh-Nagumo ODE for the given time input and model parameters.
        \dot{v}=v-v^{3}-w+R I_{\mathrm{ext}} \\
        \tau \dot{w}=v-a-b w
    
    Parameters
    ----------
    t : float
        Time input. Can also ba an array of floats.
    a : float, optional
        Model parameter. The default is -0.3.
    b : float, optional
        Model parameter. The default is 1.1.
    tau : float, optional
        Model parameter. The default is 20.
    Iext : float, optional
        Model parameter. The default is 0.23.
    x0 : list , optional
        The initial values. The default is [0, 0].

    Returns
    -------
    numpy array
        Solution.
    """
    def func(x, t):
        return np.array([x[0] - x[0] ** 3 - x[1] + Iext, (x[0] - a - b * x[1]) / tau])
    return odeint(func, x0, t)


def create_observations(data_t, data_y, geom, savename, observalbe_states=[0,1]):
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
    
    # tmp = np.vstack((data_y[idx,0], data_y[idx,1]))
    tmp = data_y[idx,observalbe_states[0]]
    print(tmp.shape, tmp.ravel().shape)
    for i in range(1,len(observalbe_states)):
        np.append(data_y[idx,observalbe_states[i]])
        
    print((idx.shape, data_t[idx].ravel().shape, data_y[idx,0].shape, data_y[idx,1].shape))
    np.savetxt(
        os.path.join(savename, "fitzhugh_nagumo_samp.dat"), np.hstack((idx, data_t[idx].ravel(), data_y[idx,0], data_y[idx,1]))
    )
    
    prt
    # Turn these timepoints into a set of points
    ptset = dde.bc.PointSet(data_t[idx])
    # Create a function that returns true when a point is part of the point set
    inside = lambda x, _: ptset.inside(x)

    # Create the observations by using the point set
    observe_y0 = dde.DirichletBC(
        geom, ptset.values_to_func(data_y[idx, 0:1]), inside, component=0
    )
    observe_y1 = dde.DirichletBC(
        geom, ptset.values_to_func(data_y[idx, 1:2]), inside, component=1
    )

    return observe_y0, observe_y1

def get_variable(v, var):
    low, up = v * 0.2, v * 1.8
    l = (up - low) / 2
    v1 = l * tf.tanh(var) + l + low
    return v1
    
    
def create_data(data_t, data_y, savename, var_trainable=[True, True, False, False], var_modifier=[-.25, 1.1, 20, 0.23], 
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
    var_list = [] # a, b, tau, Iext
    #we want to include the possibility for the variables to be both trainable and constant
    for i in range(len(var_trainable)):
        if var_trainable[i]:
            # var = scale_func(tf.Variable(0, trainable=True, dtype=tf.float32)) * var_modifier[i]
            var = tf.Variable(1e-4, trainable=True, dtype=tf.float32)
            #use 1e-4 to avoid divide by zero
            get_variable(var_modifier[i], var)
        else:
            var = tf.Variable(var_modifier[i], trainable=False, dtype=tf.float32)
        var_list.append(var)
    
    #the ode in tensorflow syntax
    def ODE(t, y):
        v1 = y[:, 0:1] - y[:, 0:1] ** 3 - y[:, 1:2] + var_list[3]
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

    observe_y0, observe_y1 = create_observations(data_t, data_y, geom, savename)
    
    data = dde.data.PDE(  
        geom,
        ODE,
        [bc0, bc1, observe_y0, observe_y1],  # list of boundary conditions
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
        layer_size=[1] + [nn_nodes]*nn_layers + [2],
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    
    def feature_transform(t):
        """
        Helper function for the feature transformation.
        """
        # t *= 1/999 #new: test if this does anything
        features = [] 
        if do_t_input_transform: #if we want to include unscaled as well
            features.append(t/999) #[0] = t
            
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
        return data_y[0] + tf.math.tanh(t) * tf.constant([1., .1]) * y
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
                batch_size=10, display_every=100, decay_amount=1e3):
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
        loss_weights=[0] * 2 + weights["bc_weights"] + weights["data_weights"],
        # loss_weights=[0] * 4 + weights["data_weights"], #try out no aux as well
        # decay=("inverse time", 10, .1),
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
        decay=("inverse time", int(sec_num_epochs), decay_amount),
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
    lr_decay,
    init_weights,
    k_vals,
    do_output_transform,
    do_t_input_transform,
    batch_size,
    true_values,
    noise,
):
    """
    This function creates a dictionary contianing all the hyperparameters, and 
    saves it to a file.
    """
    
    dictionary = dict(
        ode_weights=init_weights[0], bc_weights=init_weights[1], data_weights=init_weights[2],
        first_num_epochs=first_num_epochs, sec_num_epochs=sec_num_epochs,
        var_trainable=var_trainable, var_modifier=var_modifier, true_values=true_values,
        k_vals=k_vals, lr=lr, lr_decay=lr_decay, do_output_transform=do_output_transform,
        do_t_input_transform=do_t_input_transform, batch_size=batch_size, noise=noise,
    )

    with open(os.path.join(savename, "hyperparameters.dat"),'w') as data: 
        for key, value in dictionary.items(): 
            data.write('%s: %s\n' % (key, value))
    import pickle #try this later
    a_file = open(os.path.join(savename, "hyperparameters.pkl"), "wb") 
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
    lr=1e-2,
    init_weights = [[1, 1], [1, 1], [1, 1]],
    k_vals=[0.013],
    do_output_transform = False,
    do_t_input_transform = True,
    batch_size = 10,
    nn_layers=3,
    nn_nodes=128,
    display_every=100,
    true_values=[-.3,1.1,20,.23],
    decay_amount=1e3,
):
    """
    Function for seting up and solving the PINN.

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
   
    data, var_list = create_data(data_t, data_y, savename, var_trainable, var_modifier)

    net = create_nn(data_y, k_vals=k_vals, do_output_transform=do_output_transform, 
                    do_t_input_transform=do_t_input_transform, nn_layers=nn_layers,
                    nn_nodes=nn_nodes)
    model = dde.Model(data, net)

    callbacks = create_callbacks(var_list, savename, display_every)

    weights = default_weights(noise, init_weights)
    model_restore_path = get_model_restore_path(restore, savename)
    
    create_hyperparam_dict(savename, first_num_epochs, sec_num_epochs, var_trainable, 
                           var_modifier, lr, decay_amount, init_weights, k_vals, 
                           do_output_transform, do_t_input_transform, batch_size, 
                           true_values, noise)
    
    losshistory, train_state = train_model(
        model, weights, callbacks, first_num_epochs, sec_num_epochs, model_restore_path, lr=lr, 
        batch_size=batch_size, display_every=display_every, decay_amount=decay_amount,
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
    Solves the FHN ODE for the given timepoints, and adds noise.

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
    """ 
    Main function.
    """
    
    start = time.time()
    noise = 0.0
    savename = Path("fhn_res/fitzhugh_nagumo_res_test")
    # Create directory if not exist
    savename.mkdir(exist_ok=True)
    
    make_copy_of_program(savename)


    # a, b, tau, Iext
    a = -0.3
    b = 1.1
    tau = 20
    Iext = 0.23
    true_values = [a, b, tau, Iext]
    t_vars = [0, 999, 2000]

    t, y = generate_data(savename, true_values, t_vars, noise)
    
    var_trainable=[True, True, True, True] #a, b, tau, Iext
    
    # Train
    var_list = pinn(
        t,
        y,
        noise,
        savename,
        restore=False,
        first_num_epochs=2000,
        sec_num_epochs=int(3e5),
        var_trainable=var_trainable, #a, b, tau, Iext 
        var_modifier=[-.3, 1.1, 20., 0.23], #a, b, tau, Iext
        # init_weights = [[0, 0], [0, 0], [1, 1]], # [[ode], [bc], [data]]
        init_weights = [[20., 20.], [10., 10.], [5., 10.]], # [[ode], [bc], [data]]
        # k_vals=[0.0173], # tf.sin(k * 2*np.pi*t),
        k_vals=[.005, .01, .015, .02, .025],
        do_output_transform = True,
        do_t_input_transform = True,
        batch_size = 50,
        lr=1e-2,
        decay_amount=1e1,
        nn_layers=3,
        nn_nodes=64,
        true_values=true_values,
        display_every=1000,
    )

    # Prediction
    pred_y = fitzhugh_nagumo_model(np.ravel(t), *var_list)
    np.savetxt(
        os.path.join(savename, "fitzhugh_nagumo_pred.dat"), np.hstack((t, pred_y))
    )
    
    runtime = time.time() - start
    print("\n\nTotal runtime: {} sec".format(runtime))

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
    
    if noise>0:    
        make_plots(savename, if_noise=True, params=np.where(var_trainable)[0])
    else:
        make_plots(savename, if_noise=False, params=np.where(var_trainable)[0])
    make_one_plot(savename)
    make_samp_plot(savename)
    plot_losses(savename, do_test_vals=False)
    evaluate(savename, runtime=runtime)
    


def plot_features():

    t = np.linspace(0, 999, 1000)
    y = fitzhugh_nagumo_model(t, a = np.log(1+np.exp(0))*-.2)
    print(np.log(1+np.exp(0))*-.2)
    fig, ax = plt.subplots()
    ax.plot(t, y[:,0])
    # for i in np.linspace(-.1, -.5, 9):
    #     y = fitzhugh_nagumo_model(t, a=i)
    #     ax.plot(t, y[:,0], label=i)
    # ax.plot(t, y[:,1])
    # ax.plot(t, np.sin(0.01 * t))
    # ax.plot(t, np.sin(0.05 * t))
    # ax.plot(t, np.sin(0.1 * t))
    ax.plot(t, np.sin(0.0173*2*np.pi*t))
    ax.plot(t, np.sin(0.01*2*np.pi*t))
    ax.plot(t, np.sin(0.015*2*np.pi*t))
    ax.plot(t, np.sin(0.02*2*np.pi*t))
    # .001, .0015, .002
    
    ax.legend()
    plt.show()
    
    out = []
    for i in range(2):        
        out.append(np.mean(np.abs(y[:,i])))
    print(out)


if __name__ == "__main__":
    main()
    # plot_features()
