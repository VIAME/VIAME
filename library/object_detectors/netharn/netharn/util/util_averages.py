# -*- coding: utf-8 -*-
import collections
import ubelt as ub
import numpy as np


def _isnull(v):
    try:
        import pandas as pd
    except Exception:
        return v is None or np.isnan(v)
    else:
        return pd.isnull(v)


class MovingAve(ub.NiceRepr):
    """
    Abstract moving averages API
    """
    def average(self):
        return self.mean()

    def mean(self):
        raise NotImplementedError

    def std(self):
        raise NotImplementedError

    # def variance(self):
    #     raise NotImplementedError

    def normal(self):
        mean = self.mean()
        std = self.std()
        info = {k: {'mu': mean[k], 'sigma': std[k]}
                for k in mean.keys()}
        return info

    def update(self, other):
        raise NotImplementedError()

    def __nice__(self):
        try:
            return str(ub.urepr(self.normal(), nl=0, si=True, nobr=1, explicit=1))
        except NotImplementedError:
            return str(ub.urepr(self.average(), nl=0))

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class CumMovingAve(MovingAve):
    """
    Cumulative moving average of dictionary values

    References:
        https://en.wikipedia.org/wiki/Moving_average
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_incremental_algorithm

    TODO:
        - [X] Add support for moving variance:
            https://stackoverflow.com/questions/1174984/how-to-efficiently-calculate-a-running-standard-deviation
            https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Example:
        >>> from .util.util_averages import *
        >>> self = CumMovingAve()
        >>> print(str(self.update({'a': 10})))
        >>> print(str(self.update({'a': 0})))
        >>> print(str(self.update({'a': 2})))
        <CumMovingAve(a={mu: 10.0, sigma: 0.0})>
        <CumMovingAve(a={mu: 5.0, sigma: 5.0})>
        <CumMovingAve(a={mu: 4.0, sigma: 4.3204...})>

    Example:
        >>> from .util.util_averages import *
        >>> self = CumMovingAve(nan_method='ignore', bessel=True)
        >>> print(str(self.update({'a': 10})))
        >>> print(str(self.update({'a': 0})))
        >>> print(str(self.update({'a': np.nan})))
        <CumMovingAve(a={mu: 10.0, sigma: nan})>
        <CumMovingAve(a={mu: 5.0, sigma: 7.0710...})>
        <CumMovingAve(a={mu: 5.0, sigma: 7.0710...})>
    """
    def __init__(self, nan_method='zero', bessel=False):

        self.totals = ub.odict()
        self.weights = ub.odict()
        self.square_weights = ub.odict()
        self.means = ub.odict()
        self.var_sums = ub.odict()

        self.bessel = bessel

        self.nan_method = nan_method
        if self.nan_method not in {'ignore', 'zero'}:
            raise KeyError(self.nan_method)

    def mean(self):
        return {k: v / self.weights[k] for k, v in self.totals.items()}

    def std(self, bessel=None):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value')
            if bessel is None:
                bessel = self.bessel
            if bessel:
                return {k: np.sqrt(np.array(s) / (self.weights[k] - 1)) for k, s in self.var_sums.items()}
            else:
                return {k: np.sqrt(s / self.weights[k]) for k, s in self.var_sums.items()}

    def update(self, other):
        for k, v in other.items():
            if _isnull(v):
                if self.nan_method == 'ignore':
                    continue
                elif self.nan_method == 'zero':
                    v = 0.0
                    w = 1.0
                else:
                    raise AssertionError
            else:
                w = 1.0
            if k not in self.totals:
                self.totals[k] = 0.0
                self.weights[k] = 0.0
                self.square_weights[k] = 0.0
                self.means[k] = 0.0
                self.var_sums[k] = 0.0

            self.totals[k] += v * w
            self.weights[k] += w
            self.square_weights[k] += w ** 2
            old_mean = self.means[k]
            mean = self.means[k] = old_mean + (w / self.weights[k]) * (v - old_mean)
            delta1 = (v - old_mean)
            delta2 = (v - mean)
            self.var_sums[k] += w * delta1 * delta2
        return self


class WindowedMovingAve(MovingAve):
    """
    Windowed moving average of dictionary values

    Args:
        window (int): number of previous observations to consider

    Example:
        >>> self = WindowedMovingAve(window=3)
        >>> print(str(self.update({'a': 10})))
        >>> print(str(self.update({'a': 0})))
        >>> print(str(self.update({'a': 2})))
        <WindowedMovingAve(a={mu: 10.0, sigma: 0.0})>
        <WindowedMovingAve(a={mu: 5.0, sigma: 5.0})>
        <WindowedMovingAve(a={mu: 4.0, sigma: 4.3204...})>
    """
    def __init__(self, window=500):
        self.window = window
        self.totals = ub.odict()
        self.history = {}

    def mean(self):
        return {k: v / float(len(self.history[k])) for k, v in self.totals.items()}

    def std(self):
        # inefficient, but conforms to the api
        stds = {k: np.array(vals).std() for k, vals in self.history.items()}
        return stds

    def update(self, other):
        for k, v in other.items():
            if _isnull(v):
                v = 0
            if k not in self.totals:
                self.history[k] = collections.deque()
                self.totals[k] = 0
            self.totals[k] += v
            self.history[k].append(v)
            if len(self.history[k]) > self.window:
                # Push out the oldest value
                self.totals[k] -= self.history[k].popleft()
        return self


class ExpMovingAve(MovingAve):
    """
    Exponentially weighted moving average of dictionary values.
    Tracks multiple values which are differentiated by dictionary keys.

    Args:
        span (float): roughly corresponds to window size.
            If specified this is converted to `alpha`. Mutually exclusive with
            `alpha`. Equivalent to (2 / alpha) - 1.

        alpha (float): roughly corresponds to window size.
             Mutually exclusive with `span`.  Should range between 0 and 1.
             Equivalent to 2 / (span + 1). A higher `alpha` places more weight
             on newer examples.

        correct_bias (bool, default=False): if True, applies bias-correction
            to the final average estimation, as done in [2], otherwise
            the unmodified exponential average is returned.

    References:
        .. [1]_ http://greenteapress.com/thinkstats2/html/thinkstats2013.html
        .. [2]_ https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
        .. [3]_ https://en.wikipedia.org/wiki/Exponential_smoothing
        .. [4]_ https://en.wikipedia.org/wiki/Moving_average#Exponentially_weighted_moving_variance_and_standard_deviation

    CommandLine:
        xdoctest -m netharn.util.util_averages ExpMovingAve

    Example:
        >>> self = ExpMovingAve(span=3)
        >>> print(str(self.update({'a': 10})))
        >>> print(str(self.update({'a': 0})))
        >>> print(str(self.update({'a': 2})))
        <ExpMovingAve(a={mu: 10, sigma: 0.0})>
        <ExpMovingAve(a={mu: 5.0, sigma: 5.0})>
        <ExpMovingAve(a={mu: 3.5, sigma: 3.840...})>

    Example:
        >>> self = ExpMovingAve(span=3, correct_bias=True)
        >>> print(str(self.update({'a': 10})))
        >>> print(str(self.update({'a': 1})))
        >>> print(str(self.update({'a': 4})))
        <ExpMovingAve(a={mu: 10.0, sigma: 5.0})>
        <ExpMovingAve(a={mu: 4.0, sigma: 4.0620...})>
        <ExpMovingAve(a={mu: 4.0, sigma: 2.9154...})>
    """
    def __init__(self, span=None, alpha=None, correct_bias=False):
        self.means = ub.odict()
        self.variances = ub.odict()

        self.correct_bias = correct_bias

        # number of times we've seen each particular value
        self._n_obs = ub.odict()

        if span is None and alpha is None:
            alpha = 0
        if not bool(span is None) ^ bool(alpha is None):
            raise ValueError('specify either alpha xor span')

        if alpha is not None:
            self.alpha = alpha
        elif span is not None:
            self.alpha = 2 / (span + 1)
        else:
            raise AssertionError('impossible state')

    def mean(self):
        """
        Returns the current estimation of the moving average.

        Returns:
            Dict[str, float]: means : mapping from keys to tracked means.
        """
        if self.correct_bias:
            # Even though the examples given to the exponential moving average
            # (EMA) are "unweighted", the `alpha` means used in recursive
            # formula for the EMA do induce a weighting. This weighting
            # turns out to have a simplified form equal to
            # `1 - (1 - alpha) ** i`, where i is the number of observations.
            # When there are large numbers of observations this value goes to
            # one, which means bias correction is mainly useful when there are
            # small numbers of observations.
            beta = 1 - self.alpha
            weights = {k: (1 - beta ** i) for k, i in self._n_obs.items()}
            means =  ub.odict([(k, v / weights[k])
                                for k, v in self.means.items()])
        else:
            means = self.means
        return means

    def std(self):
        deviations =  ub.odict([(k, np.sqrt(v)) for k, v in
                                self.variances.items()])
        return deviations

    def update(self, other):
        """
        Add a new observation

        Args:
            other (Dict[str, float]): a new value for each value being tracked.
        """
        alpha = self.alpha
        for k, v in other.items():
            if _isnull(v):
                v = 0
            if self.correct_bias:
                if k not in self.means:
                    # bias correct treats values as if the initial estimate is
                    # zero
                    self.means[k] = 0
                    self.variances[k] = 0
                    self._n_obs[k] = 0

            if k not in self.means:
                # Note: We never hit this when correct_bias is False
                self.means[k] = v
                self.variances[k] = 0
                self._n_obs[k] = 1
            else:
                # Apply one step of the recursive formula for estimating the
                # new average.

                prev_mean = self.means[k]
                prev_var = self.variances[k]
                delta = v - prev_mean

                curr_mean = (alpha * v) + (1 - alpha) * prev_mean
                curr_mean_alt = alpha * delta + prev_mean
                assert np.isclose(curr_mean, curr_mean_alt)
                # print('curr_mean = {!r}'.format(curr_mean))
                # print('curr_mean_alt = {!r}'.format(curr_mean_alt))
                curr_var = (1 - alpha) * (prev_var + alpha * delta ** 2)

                self._n_obs[k] += 1
                self.means[k] = curr_mean
                self.variances[k] = curr_var
        return self


class RunningStats(ub.NiceRepr):
    """
    Dynamically records per-element array statistics and can summarized them
    per-element, across channels, or globally.

    TODO:
        - [ ] This may need a few API tweaks and good documentation
        - [ ] Move to kwarray

    SeeAlso:
        InternalRunningStats

    Example:
        >>> run = RunningStats()
        >>> ch1 = np.array([[0, 1], [3, 4]])
        >>> ch2 = np.zeros((2, 2))
        >>> img = np.dstack([ch1, ch2])
        >>> run.update(np.dstack([ch1, ch2]))
        >>> run.update(np.dstack([ch1 + 1, ch2]))
        >>> run.update(np.dstack([ch1 + 2, ch2]))
        >>> # Scalar averages
        >>> print(ub.urepr(run.simple(), nobr=1, si=True))
        >>> # Per channel averages
        >>> print(ub.urepr(ub.map_vals(lambda x: np.array(x).tolist(), run.simple()), nobr=1, si=True, nl=1))
        >>> # Per-pixel averages
        >>> print(ub.urepr(ub.map_vals(lambda x: np.array(x).tolist(), run.detail()), nobr=1, si=True, nl=1))
        """

    def __init__(run):
        run.raw_max = -np.inf
        run.raw_min = np.inf
        run.raw_total = 0
        run.raw_squares = 0
        run.n = 0

    def __nice__(self):
        return '{}'.format(self.shape)

    @property
    def shape(run):
        try:
            return run.raw_total.shape
        except Exception:
            return None

    def update(run, img):
        run.n += 1
        # Update stats across images
        run.raw_max = np.maximum(run.raw_max, img)
        run.raw_min = np.minimum(run.raw_min, img)
        run.raw_total += img
        run.raw_squares += img ** 2

    def _sumsq_std(run, total, squares, n):
        """
        Sum of squares method to compute standard deviation
        """
        numer = (n * squares - total ** 2)
        denom = (n * (n - 1.0))
        std = np.sqrt(numer / denom)
        return std

    def simple(run, axis=None):
        """
        Returns summary statistics over all cells
        """
        assert run.n > 0, 'no stats exist'
        maxi    = run.raw_max.max(axis=axis, keepdims=True)
        mini    = run.raw_min.min(axis=axis, keepdims=True)
        total   = run.raw_total.sum(axis=axis, keepdims=True)
        squares = run.raw_squares.sum(axis=axis, keepdims=True)
        if not hasattr(run.raw_total, 'shape'):
            n = run.n
        elif axis is None:
            n = run.n * np.prod(run.raw_total.shape)
        else:
            n = run.n * np.prod(np.take(run.raw_total.shape, axis))
        info = ub.odict([
            ('n', n),
            ('max', maxi),
            ('min', mini),
            ('total', total),
            ('squares', squares),
            ('mean', total / n),
            ('std', run._sumsq_std(total, squares, n)),
        ])
        return info

    def detail(run):
        """
        Returns per-cell statistics
        """
        total = run.raw_total
        squares = run.raw_squares
        maxi = run.raw_max
        mini = run.raw_min
        n = run.n
        info = ub.odict([
            ('n', n),
            ('max', maxi),
            ('min', mini),
            ('total', total),
            ('squares', squares),
            ('mean', total / n),
            ('std', run._sumsq_std(total, squares, n)),
        ])
        return info


def absdev(x, ave=np.mean, central=np.median, axis=None):
    """
    Average absolute deviation from a point of central tendency

    The `ave` absolute deviation from the `central`.

    Args:
        x (np.ndarray): input data
        axis (tuple): summarize over
        central (np.ufunc): function to get measure the center
            defaults to np.median
        ave (np.ufunc): function to average deviation over.
            defaults to np.mean

    Returns:
        np.ndarray : average_deviations

    References:
        https://en.wikipedia.org/wiki/Average_absolute_deviation

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> x = np.array([[[0, 1], [3, 4]],
        >>>               [[0, 0], [0, 0]]])
        >>> axis = (0, 1)
        >>> absdev(x, np.mean, np.median, axis=(0, 1))
        array([0.75, 1.25])
        >>> absdev(x, np.median, np.median, axis=(0, 1))
        array([0. , 0.5])
        >>> absdev(x, np.mean, np.median)
        1.0
        >>> absdev(x, np.median, np.median)
        0.0
        >>> absdev(x, np.median, np.median, axis=0)
        array([[0. , 0.5], [1.5, 2. ]])
    """
    point = central(x, axis=axis, keepdims=True)
    deviations = np.abs(x - point)
    average_deviations = ave(deviations, axis=axis)
    return average_deviations


class InternalRunningStats():
    """
    Maintains an averages of average internal statistics across a dataset.

    The difference between `RunningStats` and this is that the former can keep
    track of the average value of pixel (x, y) or channel (c) across the
    dataset, whereas this class tracks the average pixel value within an image
    across the dataset. So, this is an average of averages.

    Example:
        >>> ch1 = np.array([[0, 1], [3, 4]])
        >>> ch2 = np.zeros((2, 2))
        >>> img = np.dstack([ch1, ch2])
        >>> irun = InternalRunningStats(axis=(0, 1))
        >>> irun.update(np.dstack([ch1, ch2]))
        >>> irun.update(np.dstack([ch1 + 1, ch2]))
        >>> irun.update(np.dstack([ch1 + 2, ch2]))
        >>> # Scalar averages
        >>> print(ub.urepr(irun.info(), nobr=1, si=True))
    """

    def __init__(irun, axis=None):
        from functools import partial
        irun.axis = axis
        # Define a running stats object for each as well as the function to
        # compute the internal statistic
        irun.runs = ub.odict([
            ('mean', (
                RunningStats(), np.mean)),
            ('std', (
                RunningStats(), np.std)),
            ('median', (
                RunningStats(), np.median)),
            # ('mean_absdev_from_mean', (
            #     RunningStats(),
            #     partial(absdev, ave=np.mean, central=np.mean))),
            ('mean_absdev_from_median', (
                RunningStats(),
                partial(absdev, ave=np.mean, central=np.median))),
            ('median_absdev_from_median', (
                RunningStats(),
                partial(absdev, ave=np.median, central=np.median))),
        ])

    def update(irun, img):
        axis = irun.axis
        for run, func in irun.runs.values():
            stat = func(img, axis=axis)
            run.update(stat)

    def info(irun):
        return {
            key: run.detail() for key, (run, _) in irun.runs.items()
        }


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m netharn.util.util_averages all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
