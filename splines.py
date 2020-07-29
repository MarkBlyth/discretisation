import scipy.interpolate
import scipy.stats
import scipy.optimize
import numpy as np


def get_splinemodel_from_discretisation(discretisation, full_knots, period=None):
    """
    Given a discretisation and a full set of knots, fit a cubic
    periodic splines model, and return a function that evaluates it.
    Optionally, make the model periodic, to allow extrapolation beyond
    the base interval.

        discretisation : 1-by-n float array
            Spline discretisation, as computed by
            get_spline_discretisation_from_knots.

        full_knots : 1-by-n float array
            Position of the splines knots, as returned from
            get_spline_discretisation_from_data; (these include both
            the interior knots, and the exterior knots added to make
            the model periodic)

        period : float>0
            Optional. Period of the signal, for making an
            extrapolating model.

    Returns a function that evaluates the fitted splines model.
    """
    spline = (full_knots, discretisation, 3)

    def model(x):
        return scipy.interpolate.splev(x, spline)

    return model if period is None else make_periodic_model(model, period)


def get_spline_discretisation_from_data(data_x, data_y, full_knots):
    """
    Given some data and a set of full knots (interior and exterior),
    find the BSpline coefficients that discretise the data. data_x and
    data_y are arrays of the x and y variables of the signal, with the
    x variable relabled down to a single period. Appropriate data
    formatting can be computed with the stacker submodule.

        data_x : 1-by-n float array
            Time-like variable for the signal; rescaled to a single
            period.

        data_y : 1-by-n float array
            Dependent variable for the signal

        full_knots : 1-by-k float array
            Position of the splines knots; interior and exterior knots
            must be provided

    Returns a 1-by-k float array of the BSpline coefficients that
    discretise the data.
    """
    bspline_obj = scipy.interpolate.make_lsq_spline(data_x, data_y, full_knots)
    return bspline_obj.c


def make_periodic_model(func, period=1):
    """
    The default spline models are only accurate within the range of
    the data they were fitted to. To extrapolate periodically beyond
    the data range, the input variable must be rescaled to input
    modulo period. This function produces a extrapolating splines
    model by automating the modulo arithmetic.

        func : function
            Splines model to periodise

        period : float > 0
            Time period of the underlying signal

    Returns a periodically extrapolating splines model.
    """
    return lambda x: func(np.mod(x, period))


def get_full_knots(data_x, data_y, n_knots, n_tries=50):
    """
    Given some periodic data, find the set of interior and exterior
    knots that provide a best-possible periodic splines model to the
    data. This is done by starting with a randomly distributed set of
    knots, then attempting a numerical optimization on the knot set,
    to maximise goodness-of-fit. To avoid local minima, this procedure
    is repeated numerous times, with the best knots being recorded.

        data_x : 1-by-n float array
            Time-like variable for the signal; rescaled to a single
            period.

        data_y : 1-by-n float array
            Dependent variable for the signal

        n_knots : int > 0
            Number of knots to fit

        n_tries : int > 0
            Number of times to restart the optimisation, to avoid
            local minima. More is better, but slower.

    Returns a full knot vector (interor and exterior knots) that
    produces the lowest-residual splines fit to the provided data.
    """
    data_min, data_max = np.min(data_x), np.max(data_x)

    def loss(knotvec):
        try:
            full_spline = scipy.interpolate.splrep(data_x, data_y, t=np.sort(
                knotvec), per=True, xb=data_min, xe=data_max)

            def model(x): return scipy.interpolate.splev(x, full_spline)
            residuals = data_y - model(data_x)
            return np.linalg.norm(residuals)
        except ValueError:
            return np.inf

    initial_knots = data_min + (data_max - data_min) * \
        scipy.stats.uniform().rvs((n_tries, n_knots))

    best_loss = np.inf
    best_knots = None
    for i, k in enumerate(initial_knots):
        print("Optimisation round {0}".format(i + 1))
        opti = scipy.optimize.minimize(loss, k, tol=1e-3)
        print("Loss: ", opti.fun, opti.message)
        if opti.fun < best_loss:
            best_loss = opti.fun
            best_knots = opti.x
    full_knots, _, _ = scipy.interpolate.splrep(
        data_x, data_y, t=np.sort(best_knots), per=True, xb=data_min, xe=data_max)
    return full_knots
