import fourier
import stacker
import splines
import numpy as np
import scipy.optimize
import abc


def DEFAULT_PERIOD_ESTIMATOR(signal_y):
    """
    Default method for estimating period of a signal. This uses
    stacker.get_n_periods, with the optional arguments binded to their
    defaults. If the program ever crashes in a bizarre way, it's
    probably due to the period estimation step crashing, in which case
    this default period estimator should be replaced with something
    else.

        signal_y : 1-by-n float array
            y values of the signal to discretise. Must be sampled at
            evenly spaced points in time.

    Returns the estimated number of periodic cycles of the signal.
    """
    return stacker.get_n_periods(signal_y)


class _Discretisor(abc.ABC):
    """
    Abstract class definition for a discretisor. A discretisor must
    implement the discretise and undiscretise methods, using a signal
    that has been standardised (stacked onto unit periods) with the
    _standardisor method.
    """

    @abc.abstractmethod
    def __init__(
        self, reference_model=None, n_period_estimator=DEFAULT_PERIOD_ESTIMATOR
    ):
        """
        Parent initialiser for child classes.

            reference_model : function
                The discretisor will first align the phase of the
                signal-to-discretise, so that it matches that of a
                reference signal. The reference_model is some
                evaluable model that represents the undiscretised
                (true-period) model of a signal, as would be used for
                a control target.

            n_period_estimator : function
                The chosen method for estimating the number of periods
                in a signal. Must take a signal output (1d array of
                y-values), and return the estimated number of periods
                within that signal. The default value should work in
                most cases, but when it doesn't, an alternative method
                can be provided. Recommended alternatives are
                stacker.get_n_periods with different binded arguments,
                or stacker.acf_F0.

        """
        self.n_period_estimator = n_period_estimator
        self.reference_model = reference_model

    @abc.abstractmethod
    def discretise(self, signal):
        """
        Take a recorded signal, with evenly-spaced samples. Compute a
        discretisation of the signal, using some implementing
        discretisation method.

            signal : 2-by-n float array
                The signal to discretise. Array must be of form
                [[signal ts], [signal ys]].

        Returns discretisation, period. Period is the time taken for
        the signal to complete one cycle, discretisation is a
        unit-period representation that can be used either with the
        undiscretise method to produce a control target, or for
        continuation.
        """
        pass

    @abc.abstractmethod
    def undiscretise(self, discretisation, period):
        """
        Take a discretisation of a signal, and the period of the
        original signal. Produce an evaluable model to represent the
        signal described in the discretisation.

            discretisation : 1-by-n float array
                Some discretisation, eg. that returned by
                self.discretise, or as computed by a Newton update in
                a continuation routine.

            period : float > 0
                The desired period of the undiscretised signal.

        Returns a function of signature model(ts), which gives the
        signal value at times ts.
        """
        pass

    def update_reference_model(self, new_reference_model):
        """
        The discretisor will first align the phase of the
        signal-to-discretise, so that it matches that of a reference
        signal. reference_model is some evaluable model that
        represents the undiscretised (true-period) model of a signal,
        as would be used for a control target, that we wish to align
        the signals to.

            reference_model : function
                The new reference model to use when aligning signals.
        """
        self.reference_model = new_reference_model

    def _standardisor(self, signal):
        """
        Take a signal of arbitrary period. Align the phase of the
        signal with that of the model reference signal, and remap the
        times to a phase value in the unit interval. This allows
        discretisations to be compared across signals, regardless of
        their period or phase.

            signal : 2-by-n float array
                Signal to standardise. Of form [[signal ts],
                [signal_ys]]

        Returns a new signal array, with phase aligned to the
        reference signal, and with times remapped onto the unit
        interval; and the period of the signal.
        """
        n_periods = self.n_period_estimator(signal[1])
        period = (np.max(signal[0]) - np.min(signal[0])) / n_periods
        # If we have a reference model, align phases with it
        if self.reference_model is not None:

            def objective(phase):
                predicted_ys = self.reference_model(signal[0] - phase)
                return np.linalg.norm(signal[1] - predicted_ys)

            opti = scipy.optimize.minimise(objective, 0)
            # Rescale time to unit interval
            ts = np.mod((signal[0] - opti.x) / period, 1)
        else:
            ts = np.mod(signal[0] / period, 1)
        sort_indices = np.argsort(ts)
        return np.vstack((ts[sort_indices], signal[1][sort_indices])), period


class FourierDiscretisor(_Discretisor):
    """
    Implement a _Discretisor using a Fourier discretisation. This
    represents a periodic signal by the coefficients of its trucated
    Fourier series.
    """

    def __init__(
        self,
        n_harmonics,
        reference_model=None,
        n_period_estimator=DEFAULT_PERIOD_ESTIMATOR,
    ):
        """
        See _Discretisor.__init__ docstring.

            n_harmonics : int > 0
                Number of harmonics to use for truncated Fourier
                series.
        """
        super().__init__(reference_model, n_period_estimator)
        self.n_harmonics = n_harmonics

    def discretise(self, signal):
        signal, period = self._standardisor(signal)
        a0, ai, bi = fourier.fit_fourier_series(
            signal[0], signal[1], self.n_harmonics, 1
        )
        discretisation = np.hstack((a0, ai, bi))
        return discretisation, period

    def undiscretise(self, discretisation, period):
        a0 = discretisation[0]
        aibi = discretisation[1:].reshape((2, -1))
        return fourier.fourier_undiscretise(a0, aibi[0], aibi[1], period)


class SplinesDiscretisor(_Discretisor):
    """
    Implement a _Discretisor using a periodic splines discretisation.
    Here the discretisation is the coefficients of a fixed set of
    BSpline basis functions, as required to reconstruct the signal.
    """

    def __init__(
        self,
        full_knots,
        reference_model=None,
        n_period_estimator=DEFAULT_PERIOD_ESTIMATOR,
    ):
        """
        See _Discretisor.__init__ docstring.

            full_knots : 1-by-n array
                Set of splines knots (both interior and exterior) to
                use when fitting a splines model to the signal. Knots
                should be found using
                get_knots_for_splines_discretisor.
        """
        self.full_knots = full_knots
        super().__init__(reference_model, n_period_estimator)

    def discretise(self, signal):
        signal, period = self._standardisor(signal)
        discretisation = splines.get_spline_discretisation_from_data(
            signal[0], signal[1], self.full_knots, (0, 1)
        )
        return discretisation, period

    def undiscretise(self, discretisation, period):
        model = splines.get_splinemodel_from_discretisation(
            discretisation, self.full_knots, 1
        )
        return lambda t: model(t/period)


def get_knots_for_splines_discretisor(
    signal, n_knots, n_tries=50, n_period_estimator=DEFAULT_PERIOD_ESTIMATOR
):
    """
    Find an appropriate set of knots for a splines discretisor. The
    knots should always be found using this function, as it ensures
    the signal period is handled appropriately. The resulting knots
    can then be used for a splines discretisation of some signals of
    interest. The knots can optionally be recalculated every few
    parameter-steps, however this will change the resulting
    discretisation.

        signal : 2-by-n float array
            The signal to which knots should be fitted. Of form
            [[signal ts],[signal ys]]. Knots are fitted by finding the
            interior knots that minimise the error of a splines
            reconstruction on this signal.

        n_knots : int > 0
            Number of interior knots to fit.

        n_tries : int > 0
            Knots are selected from the best results of a series of
            randomly initialised optimisations. This specifies how
            many random initialisations to use. More is better, but
            slower.

        n_period_estimator : function
            The chosen method for estimating the number of periods
            in a signal. Must take a signal output (1d array of
            y-values), and return the estimated number of periods
            within that signal. The default value should work in
            most cases, but when it doesn't, an alternative method
            can be provided. Recommended alternatives are
            stacker.get_n_periods with different binded arguments,
            or stacker.acf_F0.

    Returns a vector of knots -- both interior and exterior -- that
    can then be passed to the initialiser of a SplinesDiscretisor.
    """
    n_periods = n_period_estimator(signal[1])
    period = (np.max(signal[0]) - np.min(signal[0])) / n_periods
    ts = np.mod(signal[0] / period, 1)
    sort_indices = np.argsort(ts)
    new_signal = np.vstack((ts[sort_indices], signal[1][sort_indices]))
    full_knots = splines.get_full_knots(
        new_signal[0], new_signal[1], n_knots, n_tries, (0, 1)
    )
    return full_knots
