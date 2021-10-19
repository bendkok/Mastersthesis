import ap_features as apf
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

np.random.seed(2)

def fitzhugh_nagumo(t, x, a, b, tau, Iext):
    """Time derivative of the Fitzhugh-Nagumo neural model.
    Parameters
    
    a = -0.3
    b = 1.4
    tau = 20
    Iext = 0.23
    
    Parameters
    ----------
    t : float
        Time (not used)
    x : np.ndarray
        State of size 2 - (Membrane potential, Recovery variable)
    a : float
        Parameter in the model
    b : float
        Parameter in the model
    tau : float
        Time scale
    Iext : float
        Constant stimulus current

    Returns
    -------
    np.ndarray
        dx/dt - size 2
    """
    return np.array([x[0] - x[0] ** 3 / 3 - x[1] + Iext, (x[0] - a - b * x[1]) / tau])


def main():

    a = -0.3
    b = 1.1
    tau = 20
    Iext = 0.23
    time = np.linspace(0, 999, 1000)
    res = solve_ivp(
        fitzhugh_nagumo,
        [0, 1000],
        [0, 0],
        args=(a, b, tau, Iext),
        t_eval=time,
    )

    v = res.y[0, :]
    w = res.y[1, :]

    s = apf.Beats(y=v, t=time)
    beats = s.chop()
    
    n = len(time)
    idx = np.random.choice(np.arange(1, n - 1), size=n // 4, replace=False)
    # Add the last point to the list
    idx = np.append(idx, [0, n - 1])

    # Plot al beats
    fig, ax = plt.subplots()
    # ax.plot(time[idx], v[idx], "o", label="$v$")
    ax.plot(time, v, label="$v$")
    # ax.plot(time, w, label="$w$")
    ax.plot(time, 2*np.sin(2*np.pi * time * 0.016))
    ax.legend()
    ax.set_xlabel("Time [ms]")
    ax.set_title("Fithugh Nagumo model")

    fig, ax = plt.subplots()
    for beat in beats:
        ax.plot(beat.t, beat.y)
    ax.plot(time, 2*np.sin(2*np.pi * time * 0.016))
    ax.set_title("Chopped beats, n = {}".format(len(beats)))

    # Plot some APDs
    fig, ax = plt.subplots()
    apds = [20, 40, 50, 70, 80]
    N = len(apds)
    x = np.arange(N)
    width = 1 / (s.num_beats + 1)
    for i, beat in enumerate(beats):
        ax.bar(x + i * width, [beat.apd(apd) for apd in apds], width=width)
    ax.set_xticks(x + 0.5 - width)
    ax.set_xticklabels(apds)
    ax.set_ylabel("Time [ms]")
    ax.set_xlabel("APD")
    ax.grid()
    ax.set_title("Action potential duration for different beats")
    plt.show()


if __name__ == "__main__":
    main()
