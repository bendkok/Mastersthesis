from scipy.optimize import minimize, brute, basinhopping, differential_evolution
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

from fitzhugh_nagumo import fitzhugh_nagumo as rhs


def solve(t, params, y0=None):
    if y0 is None:
        y0 = [0, 0]
    res = solve_ivp(
        rhs,
        [t[0], t[-1]],
        y0,
        args=params,
        t_eval=t,
    )
    return res.y


def optimize_all_parameters():

    # Generate data
    t = np.linspace(0, 999, 1000)

    a = -0.3
    b = 1.4
    tau = 20
    Iext = 0.23
    params = (a, b, tau, Iext)
    y_data = solve(t, params)

    def fun(x):
        y = solve(t, x)
        return np.linalg.norm(y - y_data) ** 2

    x0 = (-0.1, 1.0, 15, 0.2)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    res = minimize(fun, x0, method="Nelder-Mead", tol=1e-6)

    print(res)


def optimize_two_parameters():

    method = "differential_evolution"

    # Generate data
    t = np.linspace(0, 999, 1000)

    a = -0.3
    b = 1.4
    tau = 20
    Iext = 0.23
    true_params = (a, b, tau, Iext)
    y_data = solve(t, true_params)

    def fun(x):
        params = (x[0], x[1], tau, Iext)
        y = solve(t, params)
        err = np.linalg.norm(y[0] - y_data[0]) ** 2
        # err = np.linalg.norm(y - y_data) ** 2
        print(params, err)
        return err

    x0 = (-0.2, 1.1)
    y_initial = solve(t, (x0[0], x0[1], tau, Iext))

    if method == "nelder-mead":
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        # res = minimize(fun, x0, method="BFGS", tol=1e-6)
        res = minimize(
            fun, x0, method="Nelder-Mead", tol=1e-6, options={"maxiter": 100}
        )
        x = res.x
        print(res)
    elif method == "brute":
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html
        x = brute(fun, ranges=((-0.5, 0), (1, 1.7)))

    elif method == "differential_evolution":
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
        res = differential_evolution(fun, bounds=((-0.5, 0), (1, 1.7)), tol=0.02, atol=.01)
        x = res.x
        print(res)
    else:
        raise ValueError(f"Unkown method {method}")

    print(f"True value: {a, b}")
    print(f"Predicted value: {x}")
    y_pred = solve(t, (x[0], x[1], tau, Iext))

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, y_data[0], label="data")
    ax[0].plot(t, y_initial[0], label="initial")
    ax[0].plot(t, y_pred[0], label="predicted")
    ax[0].set_title("$v$")
    ax[1].plot(t, y_data[1], label="data")
    ax[1].plot(t, y_initial[1], label="initial")
    ax[1].plot(t, y_pred[1], label="predicted")
    ax[1].set_title("$w$")
    for axi in ax:
        axi.legend()
        axi.grid()
    plt.show()


if __name__ == "__main__":
    optimize_two_parameters()
