import numpy as np
import warnings
import statsmodels.tsa.stattools as stattools
import sys
import os

sys.path.append(os.path.dirname(
    os.path.realpath(__file__)) + "/fastF0Nls/python")
import single_pitch


class NotEnoughPeriodsError(Exception):
    pass


def autocorrelation_F0_estimate(signal_y, FFT=False):
    """Estimate the number of cycles in a given signal from the first
    zero of the autocorrelation function. The autocorrelation function
    acf(X,n) gives the correlation between time series entry x_i, and
    time series entry x_{i+n}. As a periodic signal will fully
    correlate with itself one period ago, we can use this to estimate
    the fundamental frequency. The first zero should occur at one
    quarter of the signal's period, so this is found and used as an
    estimate of F0. Note that this requires the data to be sampled at
    consistent time intervals!

        signal_y : float array
            1d array of time series data, containing the signal
            that we wish to estimate the fundamental frequency of

        FFT : bool
            Whether or not to use FFT for calculating the
            autocorrelation. Faster for long time series data.

    Raises the following:

        NotEnoughPeriodsError : the data does not contain enough
                                periods to accurately estimate the
                                fundamental frequency

        ValueError : raised when signal_y cannot be cast to a 1d
                    array

    Returns a float, representing the number of cycles estimated to
    take place across the signal.
    """
    ts_data = np.array(signal_y, dtype=float)
    if not len(ts_data.shape) == 1:
        raise ValueError(
            "signal_y must be a 1-dimensional array-like object")
    # Start off with just one lag
    n_lags = 1
    contains_zero = False
    # While the ACF up to n_lags doesn't cross zero...
    while not contains_zero:
        # Find the ACF
        acf = stattools.acf(ts_data, nlags=n_lags, fft=FFT)
        # See if it crossees zero
        is_crossing_point = (acf[:-1] * acf[1:]) < 0
        contains_zero = np.any(is_crossing_point)
        # If we've already checked max. possible lags, thow error
        if (not contains_zero) and (n_lags == ts_data.shape[0] - 1):
            raise NotEnoughPeriodsError(
                "No zero found in autocorrelation function")
        # Check more lags
        n_lags = min(ts_data.shape[0] - 1, n_lags * 2)
    first_zero = np.where(is_crossing_point)[0][0]
    return ts_data.shape[0] / (4 * (first_zero + 1))


def fast_nls_F0_estimate(signal_y, estimated_cycles, n_harmonics=10):
    """Estimate the number of cycles in a given signal, using the
    fastNLS algorithm. Gives a higher accuracy and more robustness to
    noise than correlation-based methods, at the expense of requiring
    a reasonable fundamental frequency estimate to work from.

        signal_y : float array
            1d array of time series data, containing the signal
            that we wish to estimate the fundamental frequency of

        estimated_cycles : 2-element array-like object
            Lower and upper bounds on the number of cycles the signal
            contains

        n_harmonics : int
            Number of harmonics to use for fundamental frequency
            estimation.

    Raises the following:

        ValueError : estimated_cycles cannot be cast as a 2-element
                    float array

        ValueError : raised when signal_y cannot be cast to a 1d
                    array

        UserWarning : the signal is estimated to contain fewer than
                    4 full cycles; using too few cycles may result in
                    an unreliable F0 estimate

    Returns the estimated number of cycles in the given signal.

    NOTE: this only uses the first zero of the ACF, which should
    hopefully improve efficiency. For a more accurate approach under
    noise, the location of all zeros could be used.
    """
    # Check estimated cycles
    try:
        f0_bounds = np.array(estimated_cycles, dtype=float).reshape((2,))
    except ValueError:
        raise ValueError(
            "estimated_cycles must be a float-array-like object of form [lower bound, upper bound]"
        )
    if estimated_cycles[0] < 4:
        warnings.warn(
            "Period estimation can become unreliable on data containing few cycles"
        )
    # Check signal_y
    ts_data = np.array(signal_y)
    if not len(ts_data.shape) == 1:
        raise ValueError(
            "signal_y must be a 1-dimensional array-like object")
    # Estimate fundamental frequency
    sampling_freq = ts_data.shape[0]
    bounds = f0_bounds / (2 * sampling_freq)
    f0_estimator = single_pitch.single_pitch(
        sampling_freq, n_harmonics, bounds)
    estimate = (sampling_freq / np.pi) * f0_estimator.est(ts_data)
    return estimate


def get_n_periods(signal_y, padding=20, n_harmonics=20, FFT=True, skip_NLS=False):
    """Estimate the number of cycles in a periodic time series. First
    uses autocorrelation methods to get a loose estimate of the
    answer; this is then refined using a fast nonlinear least squares
    method.

        signal_y : float array
            1d array of time series data, containing the signal
            that we wish to estimate the fundamental frequency of

        padding : float
            Bounds must be specified for the NLS F0 estimation. These
            bounds are given by the acf estimate +- padding %. Large
            paddings indicate a low degree of confidence in the acf
            results.

        n_harmonics : int
            Number of harmonics to use for NLS F0 estimation

        FFT : bool
            Whether or not to use the FFT method for calculating
            autocorrelation. Should be set to true for efficient
            computation on large numbers of datapoints.

        skip_NLS : bool
            Estimate period only using autocorrelation.

    Raises the following:

        ValueError : raised when signal_y cannot be cast to a 1d
                    array

        ValueError : raised when padding is not between 0 and 100

        ValueError : raised when a non-positive number of harmonics is
                    specified

        TypeError : if n_harmonics is not an int

        Passes up any other exceptions.

    Returns a float estimating the number of periods in the signal.
    """
    # Check signal_y
    ts_data = np.array(signal_y)
    if not len(ts_data.shape) == 1:
        raise ValueError(
            "signal_y must be a 1-dimensional array-like object")
    if not 0 < padding <= 100:
        raise ValueError("padding must be between 0 and 100")
    if not n_harmonics > 0:
        raise ValueError("n_harmonics  must be greater than zero")
    if not isinstance(n_harmonics, int):
        raise TypeError("n_harmonics must be int")
    # Form an estimate of the fundamental frequency using autocorrelation
    acf_F0 = autocorrelation_F0_estimate(signal_y, FFT)
    if skip_NLS:
        return acf_F0
    # Use this estimate to form bounds for NLS method
    bounds = np.array(
        [acf_F0 - padding / 100 * acf_F0, acf_F0 + padding / 100 * acf_F0]
    )
    nls_F0 = fast_nls_F0_estimate(ts_data, bounds, n_harmonics)
    return nls_F0


def stack_periods(
    signal_y,
    t_range=None,
    sorted=False,
    padding=20,
    n_harmonics=20,
    FFT=True,
    skip_NLS=False,
):
    """Take a periodic signal of unknown period, with evenly sampled
    datapoints. Compute the period of the signal, and produce a
    transformed set of times, such that the new times of any given
    datapoint all lie within the same period. This effectively takes
    each cycle and stacks them on top of each other.

        signal_y : float array
            1d array of time series data, containing the signal
            that we wish to estimate the fundamental frequency of

        t_range : float
            Time-range over which signal_y was recorded. If None,
            time is rescaled to the unit interval. If set, time is
            retained as period time.

        sorted : bool
            Whether or not to sort the resulting t, signal_y
            arrays.

        padding : float
            Bounds must be specified for the NLS F0 estimation. These
            bounds are given by the acf estimate +- padding %. Large
            paddings indicate a low degree of confidence in the acf
            results.

        n_harmonics : int
            Number of harmonics to use for NLS F0 estimation

        FFT : bool
            Whether or not to use the FFT method for calculating
            autocorrelation. Should be set to true for efficient
            computation on large numbers of datapoints.

        skip_NLS : bool
            Estimate period only using autocorrelation.

    Passes up the following:

        UserWarning : the signal is estimated to contain fewer than
                    4 full cycles; using too few cycles may result in
                    an unreliable F0 estimate

        ValueError : raised when signal_y cannot be cast to a 1d
                    array

        ValueError : raised when padding is not between 0 and 100

        ValueError : raised when a non-positive number of harmonics is
                    specified

        TypeError : if n_harmonics is not an int

    Returns signal_y, ts for the stacked signal. signal_y
    remains unchanged if sorted is False.
    """
    n_periods = get_n_periods(
        signal_y, padding, n_harmonics, FFT, skip_NLS)
    if t_range is None:
        ts = np.mod(np.linspace(0, n_periods, len(signal_y)), 1)
    else:
        ts = np.mod(np.linspace(0, t_range, len(
            signal_y)), t_range / n_periods)
    if not sorted:
        return ts, signal_y
    sort_indices = np.argsort(ts)
    return ts[sort_indices], signal_y[sort_indices]
