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
import matplotlib.pyplot as plt

from postprocessing import saveplot


def fitzhugh_nagumo_model(
    t, a=-0.3, b=1.4, tau=20, Iext=0.23, x0 = [0, 0]   # maybe try different init values
):
    def func(x, t):
        #shouldn't v^3 be divided by 3?
        return np.array([x[0] - x[0] ** 3 - x[1] + Iext, (x[0] - a - b * x[1]) / tau])

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


def create_data(data_t, data_y):

    # Define the variables in the model
    a = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) * 0.1
    b = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32))
    tau = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) * 10
    Iext = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) * 0.1
    var_list = [a, b, tau, Iext]

    def ODE(t, y):
        v1 = y[:, 0:1] - y[:, 0:1] ** 3 - y[:, 1:2] + Iext
        v2 = (y[:, 0:1] - a - b * y[:, 1:2]) / tau
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

    data = dde.data.PDE(  # should this be TimePDE?
        geom,
        ODE,
        [bc0, bc1, observe_y4, observe_y5],  # list of boundary conditions
        anchors=data_t,
    )
    return data, var_list


def create_nn(data_y):
    # Feed-forward neural networks
    net = dde.maps.FNN(
        layer_size=[1, 128, 128, 128, 2],
        activation="swish",
        kernel_initializer="Glorot normal",
    )

    def feature_transform(t):
        return tf.concat(
            (
                t,
                tf.sin(0.01 * t),
                tf.sin(0.05 * t),
                tf.sin(0.1 * t),
                tf.sin(0.15 * t),
            ),
            axis=1,
        )

    net.apply_feature_transform(feature_transform)

    def output_transform(t, y):
        # Weights in the output layer are chosen as the magnitudes
        # of the mean values of the ODE solution
        return data_y[0] + tf.math.tanh(t) * tf.constant([0.1, 0.1]) * y

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
    # Save variables after 1000 ephocs
    variable = dde.callbacks.VariableValue(
        var_list,
        period=1000,
        filename=os.path.join(savename, "variables.dat"),
        precision=3,
    )
    return [checkpointer, variable]


def default_weights(noise):
    bc_weights = [1, 1]
    if noise >= 0.1:
        bc_weights = [w * 10 for w in bc_weights]

    data_weights = [1, 1]
    # Large noise requires small data_weights
    if noise >= 0.1:
        data_weights = [w / 10 for w in data_weights]

    ode_weights = [1, 1]
    # Large noise requires large ode_weights
    if noise > 0:
        ode_weights = [10 * w for w in ode_weights]

    return dict(
        bc_weights=bc_weights, data_weights=data_weights, ode_weights=ode_weights
    )


def train_model(model, weights, callbacks, first_num_epochs, sec_num_epochs, model_restore_path=None):

    # First compile the model with ode weights set to zero
    model.compile(
        "adam",
        lr=1e-3,
        loss_weights=[0] * 2 + weights["bc_weights"] + weights["data_weights"],
    )
    # And train
    model.train(epochs=int(first_num_epochs), display_every=1000)
    
    # Now compile the model, but this time include the ode weights
    model.compile(
        "adam",
        lr=1e-3,
        loss_weights=weights["ode_weights"]
        + weights["bc_weights"]
        + weights["data_weights"],
    )

    losshistory, train_state = model.train(
        epochs=int(sec_num_epochs),
        display_every=1000,
        callbacks=callbacks,
        disregard_previous_best=True,
        model_restore_path=model_restore_path,
    )
    return losshistory, train_state


def pinn(
    data_t,
    data_y,
    noise,
    savename,
    restore=False,
    first_num_epochs=int(1e3),
    sec_num_epochs=int(1e5),
):
    """
    Parameters
    ----------
    data_t : TYPE
        DESCRIPTION.
    data_y : TYPE
        DESCRIPTION.
    noise : TYPE
        DESCRIPTION.
    savename : TYPE
        DESCRIPTION.
    restore : TYPE, optional
        DESCRIPTION. The default is False.
    first_num_epochs : TYPE, optional
        DESCRIPTION. The default is int(1e3).
    sec_num_epochs : TYPE, optional
        DESCRIPTION. The default is int(1e5).

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    data, var_list = create_data(data_t, data_y)

    net = create_nn(data_y)
    model = dde.Model(data, net)

    callbacks = create_callbacks(var_list, savename)

    weights = default_weights(noise)
    #if you want to restore from a previous run
    if restore == True:
        #reads form the checkpoint text file
        with open(os.path.join(savename,"model/checkpoint"), 'r') as reader:
            inp = reader.read()
            restore_from = inp.split(" ")[1].split('"')[1]
        model_restore_path=os.path.join(savename,"model", restore_from)
    else:
        model_restore_path = None
        
    losshistory, train_state = train_model(
        model, weights, callbacks, first_num_epochs, sec_num_epochs, model_restore_path
    )

    saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=savename)

    var_list = [model.sess.run(v) for v in var_list]
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


def main():
    start = time.time()
    noise = 0.0
    # tf.device("gpu")
    savename = Path("fitzhugh_nagumo_res")
    # Create directory if not exist
    savename.mkdir(exist_ok=True)

    # a, b, tau, Iext
    a = -0.3
    b = 1.4
    tau = 20
    Iext = 0.23
    true_values = [a, b, tau, Iext]
    t_vars = [0, 999, 1000]

    t, y = generate_data(savename, true_values, t_vars, noise)

    # Train
    var_list = pinn(
        t,
        y,
        noise,
        savename,
        restore=False,
        first_num_epochs=1000,
        sec_num_epochs=90000,
    )

    # Prediction
    pred_y = fitzhugh_nagumo_model(np.ravel(t), *var_list)
    np.savetxt(
        os.path.join(savename, "fitzhugh_nagumo_pred.dat"), np.hstack((t, pred_y))
    )
    print("\n\nTotal runtime: {}".format(time.time() - start))

    print("Original values: ")
    print(f"(a, b tau, Iext) = {true_values=}")

    print("Predicted values: ")
    print(f"(a, b tau, Iext) = {var_list=}")

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


def plot_features():

    t = np.linspace(0, 999, 1000)
    y = fitzhugh_nagumo_model(t)
    fig, ax = plt.subplots()
    ax.plot(t, y)
    ax.plot(t, np.sin(0.01 * t))
    ax.plot(t, np.sin(0.05 * t))
    ax.plot(t, np.sin(0.1 * t))
    ax.plot(t, np.sin(0.15 * t))

    plt.show()


if __name__ == "__main__":
    main()
    plot_features()
