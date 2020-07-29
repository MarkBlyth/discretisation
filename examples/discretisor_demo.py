#!/usr/bin/env python3

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
# Only use like this from within the project directory; otherwise just
# import discretisation
import discretisors

T_MAX = 100
N_POINTS = 500
NOISE = 0.1
N_KNOTS = 8
N_HARMONICS = 20


def fitzhugh_nagumo(x, I=1):
    """
    Defines the RHS of the FH model
        x: ndarray
            np array of the system state, of form [v, w]

        **kwargs:
            Optional parameters. The parameters that can be modified,
            and their default values, are as follows:
                I : 0; applied current

    Returns the derivatives [v_dot, w_dot]
    """
    v, w = x  # Unpack state
    v_dot = v - (v ** 3) / 3 - w + I
    w_dot = 0.08 * (v + 0.7 - 0.8 * w)
    return np.array([v_dot, w_dot])


# Generate a signal by solving the FH system
ts = np.linspace(0, T_MAX, N_POINTS)
solution = scipy.integrate.solve_ivp(
    lambda t, x: fitzhugh_nagumo(x),
    t_span=[-50, T_MAX],
    y0=[0, 0],
    t_eval=ts,
    rtol=1e-6,
)
ys_clean = solution.y[0]
# Noise-corrupt the signal
ys_noisy = ys_clean + np.random.normal(0, NOISE, ys_clean.shape)
signal = np.vstack((ts, ys_noisy))

# Build a splines discretisor from the knots
knots = discretisors.get_knots_for_splines_discretisor(signal, N_KNOTS)
spline_discretisor = discretisors.SplinesDiscretisor(knots)
splines_discretisation, period = spline_discretisor.discretise(signal)
splines_model = spline_discretisor.undiscretise(splines_discretisation, period)

# Build a Fourier discretisor
fourier_discretisor = discretisors.FourierDiscretisor(N_HARMONICS)
fourier_discretisation, period = fourier_discretisor.discretise(signal)
fourier_model = fourier_discretisor.undiscretise(
    fourier_discretisation, period)


# Plot!
fig, axarr = plt.subplots(2)
axarr[0].plot(ts, ys_noisy, label="Training signal")
axarr[1].plot(ts, ys_clean, color="k", linestyle="--", label="Latent signal")
axarr[1].plot(ts, splines_model(ts), color="red", label="Splines model")
axarr[1].plot(ts, fourier_model(ts), color="blue", label="Fourier model")
axarr[0].legend()
axarr[1].legend()
plt.tight_layout()
plt.show()
