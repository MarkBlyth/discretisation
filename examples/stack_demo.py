#!/usr/bin/env python3

import numpy as np
import stacker
import scipy.integrate
import matplotlib.pyplot as plt

T_MAX = 150
N_POINTS = 500
NOISE = 0.1


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
ys = solution.y[0]

# Noise-corrupt the signal
ys += np.random.normal(0, NOISE, ys.shape)

# Stack the periods
stacked_ts, stacked_ys = stacker.stack_periods(ys, t_range=T_MAX)

# Or, rescale to t \in [0, 1], by not specifying the signal time range
# stacked_ts, stacked_ys = stacker.stack_periods(ys)

# Plot
fig, axarr = plt.subplots(2)
axarr[0].plot(ts, ys)
axarr[1].scatter(stacked_ts, stacked_ys)
plt.show()
