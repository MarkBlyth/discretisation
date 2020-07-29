
# Table of Contents

1.  [About](#org92bab6a)
2.  [Discretisors](#org7d8d01c)
3.  [Fourier](#org15264d2)
4.  [Splines](#orgcb25a65)


<a id="org92bab6a"></a>

# About

Simple module to discretise periodic signals.
See the examples file for demos of module usage, and the function docstrings for documentation on how to use them and what they do.


<a id="org7d8d01c"></a>

# Discretisors

The standard interface to the discretisation methods provided here.
Discretisor objects provide a standardised method for discretising and
undiscretising signals, using various discretisation methods.

A signal is a 2d array of form 

    [[t0, t1, ..., tn], [y0, y1, ..., yn]]

Given a signal, the discretisors can be used as follows.

Splines discretisation:

    knots = discretisors.get_knots_for_splines_discretisor(signal, n_knots)
    discretisor = discretisors.SplinesDiscretisor(knots)

Fourier discretisation:

    discretisor = discretisors.FourierDiscretisor(n_harmonics)

Signals can then be discretised and undiscretised like so:

    discretisation, period = discretisor.discretise(signal)
    model = discretisor.undiscretise(discretisation, period)


<a id="org15264d2"></a>

# Fourier

-   Fit a truncated Fourier series
-   Evaluate a truncated Fourier series


<a id="orgcb25a65"></a>

# Splines

-   Fit a splines discretisation to data
-   Build an evaluable splines model from a discretisation
-   Make a fitted splines model periodic
-   Fit a set of interior knots

