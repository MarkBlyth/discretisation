#!/usr/bin/env python3

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

# Only use like this from within the project directory
import splines
import stacker

T_MAX = 100
N_POINTS = 500
NOISE = 0.1
N_KNOTS = 8

# Define a model, so that we have some signal to work with
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


# Generate a signal (by solving the FH system)
ts = np.linspace(0, T_MAX, N_POINTS)
solution = scipy.integrate.solve_ivp(
    lambda t, x: fitzhugh_nagumo(x),
    t_span=[-50, T_MAX],
    y0=[0, 0],
    t_eval=ts,
    rtol=1e-6,
)
clean_ys = solution.y[0]

# Noise-corrupt the signal
ys = clean_ys + np.random.normal(0, NOISE, clean_ys.shape)

# Stack the periods
stacked_ts, stacked_ys = stacker.stack_periods(ys, t_range=T_MAX, sorted=True)

# Optimise the interior knots
opti_knots = splines.get_interior_knots(
    stacked_ts, stacked_ys, n_knots=N_KNOTS, n_tries=50
)

# Find a discretisation and build a model from that. This produces a
# new set of knots, which contain both the optimised interior knots,
# and the necessary endpoint knots required to build a periodic
# splines model.
full_knots, discretisation = splines.get_spline_discretisation_from_data(
    stacked_ts, stacked_ys, opti_knots
)
model = splines.get_splinemodel_from_discretisation(discretisation, full_knots)

# Make the model periodic
period = np.max(stacked_ts)
periodic_model = splines.make_periodic_model(model, period=period)

# Plot!
fig, axarr = plt.subplots(2)
axarr[0].plot(ts, ys, label="Training signal")
axarr[1].scatter(stacked_ts, stacked_ys, label="Stacked periods")
axarr[1].plot(ts, clean_ys, color="k", linestyle="--", label="Latent signal")
axarr[1].plot(ts, periodic_model(ts), color="red", label="Splines model")
axarr[0].legend()
axarr[1].legend()
plt.tight_layout()
plt.show()
