import numpy as np


def fit_fourier_series(data_t, data_y, n_harmonics, period):
    """
    Given a set of datapoints, fit a truncated Fourier series of given
    period.

        data_t : 1-by-n array
            Time points for the signal to fit

        data_y : 1-by-n array
            Signal value at each time point

        n_harmonics : int > 0
             Order of the truncated Fourier series to fit

        period : float > 0
             Time taken for the signal to complete a single
             oscillation

    Returns a tuple (a0, ai, bi), for the DC offset, cosine
    coefficients, and sine coefficients of the fitted truncated
    Fourier series.
    """
    data_t = np.array(data_t, dtype=float).reshape((-1,))
    ones_mat = np.ones((n_harmonics, len(data_t)))
    # Transpose of trig args in eval function
    # Matrix M[r,c] = (c+1) * ts[r]; cols give harmonics, rows give ts
    trig_args = (
        2
        * np.pi
        / period
        * (np.arange(1, n_harmonics + 1).reshape((-1, 1)) * ones_mat).T
        * data_t.reshape((-1, 1))
    )
    one = np.ones((len(data_t), 1))
    design_mat = np.hstack((one, np.cos(trig_args), np.sin(trig_args)))
    lsq_solution = np.linalg.lstsq(design_mat, data_y.reshape((-1, 1)))
    lsqfit = lsq_solution[0].reshape((-1,))
    a0, ai, bi = lsqfit[0], lsqfit[1: n_harmonics +
                                   1], lsqfit[n_harmonics + 1:]
    return a0, ai, bi


def fourier_undiscretise(a0, ai, bi, period):
    """
    Produce a model to evaluate a truncated Fourier series.

        a0 : float
            DC offset

        ai : 1-by-n array
            Cosine harmonics coefficients

        bi : 1-by-n array 
            Sine harmonics coefficients

        period : float > 0
            Time taken for the signal to complete a single oscillation

    Returns a function that evaluates the truncated Fourier series at
    a 1-by-n array of time points.
    """
    ai, bi = ai.reshape((1, -1)), bi.reshape((1, -1))
    n_harmonics = ai.shape[1]

    def deparameterized(ts):
        ts = np.array(ts, dtype=float).reshape((-1,))
        # Transpose of fitting trig args
        ones_mat = np.ones((len(ts), n_harmonics))
        trig_args = (
            2
            * np.pi
            / period
            * (ts.reshape((-1, 1)) * ones_mat).T
            * np.arange(1, n_harmonics + 1).reshape((-1, 1))
        )
        coss = np.matmul(ai, np.cos(trig_args))
        sins = np.matmul(bi, np.sin(trig_args))
        return np.squeeze(a0 + sins + coss)

    return deparameterized
