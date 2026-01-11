"""
Class for monitoring performance on validation data.

TODO:
    - [ ] Implement algorithm from dlib
    http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html
"""
from netharn import util
import itertools as it
import numpy as np
import ubelt as ub

__all__ = ['Monitor']


def demodata_monitor(ignore_first_epochs=0):
    rng = np.random.RandomState(0)
    n = 300
    losses = (sorted(rng.randint(10, n, size=n)) + rng.randint(0, 20, size=n) - 10)[::-1]
    mious = (sorted(rng.randint(10, n, size=n)) + rng.randint(0, 20, size=n) - 10)
    monitor = Monitor(minimize=['loss'], maximize=['miou'], smoothing=0.0,
                      ignore_first_epochs=ignore_first_epochs)
    for epoch, (loss, miou) in enumerate(zip(losses, mious)):
        monitor.update(epoch, {'loss': loss, 'miou': miou})
    return monitor


class Monitor(ub.NiceRepr):
    """
    Monitors an instance of FitHarn as it trains. Makes sure that measurements
    of quality (e.g. loss, accuracy, AUC, mAP, etc...) on the validation
    dataset continues to go do (or at least isn't increasing), and stops
    training early if certain conditions are met.

    Attributes:
        minimize (List[str]): measures where a lower is better
        maximize (List[str]): measures where a higher is better
        smoothing (float, default=0.0): smoothness factor for moving averages.
        max_epoch (int, default=1000): number of epochs to stop after
        patience (int, default=None): if specified, the number of epochs
            to wait before quiting if the quality metrics are not improving.
        min_lr (float): If specified stop learning after lr drops beyond this
            point
        ignore_first_epochs (int): If specified, ignore the results from the
            first few epochs. Determine what the best model is after this
            point.

    Example:
        >>> # simulate loss going down and then overfitting
        >>> from .monitor import *
        >>> rng = np.random.RandomState(0)
        >>> n = 300
        >>> losses = (sorted(rng.randint(10, n, size=n)) + rng.randint(0, 20, size=n) - 10)[::-1]
        >>> mious = (sorted(rng.randint(10, n, size=n)) + rng.randint(0, 20, size=n) - 10)
        >>> monitor = Monitor(minimize=['loss'], maximize=['miou'], smoothing=.6)
        >>> for epoch, (loss, miou) in enumerate(zip(losses, mious)):
        >>>     monitor.update(epoch, {'loss': loss, 'miou': miou})
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> monitor.show()

    Example:
        >>> # Test the ignore first param
        >>> from .monitor import *
        >>> rng = np.random.RandomState(0)
        >>> n = 300
        >>> losses = (sorted(rng.randint(10, n, size=n)) + rng.randint(0, 20, size=n) - 10)[::-1]
        >>> mious = (sorted(rng.randint(10, n, size=n)) + rng.randint(0, 20, size=n) - 10)
        >>> monitor = Monitor(minimize=['loss'], smoothing=.6, ignore_first_epochs=3)
        >>> monitor.update(0, {'loss': 0.001})
        >>> monitor.update(1, {'loss': 9.40})
        >>> monitor.update(2, {'loss': 1.40})
        >>> monitor.update(3, {'loss': 0.40})
        >>> monitor.update(4, {'loss': 0.30})
        >>> monitor.update(5, {'loss': 0.35})
        >>> monitor.update(6, {'loss': 0.33})
        >>> monitor.update(7, {'loss': 0.31})
        >>> monitor.update(8, {'loss': 0.32})
        >>> monitor.update(9, {'loss': 0.33})
        >>> monitor.update(10, {'loss': 0.311})
        >>> monitor.update(11, {'loss': 0.4})
        >>> monitor.update(12, {'loss': 0.5})
        >>> monitor.update(13, {'loss': 0.6})
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> monitor.show()
    """

    def __init__(monitor, minimize=['loss'], maximize=[], smoothing=0.0,
                 patience=None, max_epoch=1000, min_lr=None, ignore_first_epochs=0):

        # Internal attributes
        monitor._ewma = util.ExpMovingAve(alpha=1 - smoothing)
        monitor._raw_metrics = []
        monitor._smooth_metrics = []
        monitor._epochs = []
        monitor._is_good = []

        # Bookkeeping
        monitor._current_lr = None
        monitor._current_epoch = None
        monitor._best_raw_metrics = None
        monitor._best_smooth_metrics = None
        monitor._best_epoch = None
        monitor._n_bad_epochs = 0

        # Keep track of which metrics we want to maximize / minimize
        monitor.minimize = minimize
        monitor.maximize = maximize

        # early stopping
        monitor.patience = patience
        monitor.max_epoch = max_epoch
        monitor.min_lr = min_lr
        monitor.ignore_first_epochs = ignore_first_epochs

    def __nice__(self):
        import ubelt as ub
        return ub.repr2({
            'patience': self.patience,
            'max_epoch': self.max_epoch,
            'min_lr': self.min_lr,
            'ignore_first_epochs': self.ignore_first_epochs,
        }, nl=0)

    @classmethod
    def coerce(cls, config, **kw):
        """
        Coerce args to create a Monitor object from a configuration dictionary.
        Accepts keywords 'max_epoch', 'patience', and 'min_lr'.

        Returns:
            Tuple[type, dict]: returns the monitor class and its initkw.

        Example:
            >>> config = {'min_lr': 1e-5}
            >>> cls, initkw = Monitor.coerce(config)
            >>> print('initkw = {}'.format(ub.repr2(initkw, nl=1)))
            initkw = {
                'ignore_first_epochs': 0,
                'max_epoch': 100,
                'min_lr': 1e-05,
                'minimize': ['loss'],
                'patience': 100,
            }
        """
        from .api import _update_defaults
        config = _update_defaults(config, kw)
        max_epoch = config.get('max_epoch', 100)
        return (cls, {
            'minimize': ['loss'],
            'max_epoch': max_epoch,
            'patience': config.get('patience', max_epoch),
            'min_lr': config.get('min_lr', None),
            'ignore_first_epochs': config.get('ignore_first_epochs', 0),
        })

    def show(monitor):
        """
        Draws the monitored metrics using matplotlib
        """
        import matplotlib.pyplot as plt
        import kwplot
        import pandas as pd
        smooth_ydatas = pd.DataFrame.from_dict(monitor._smooth_metrics).to_dict('list')
        raw_ydatas = pd.DataFrame.from_dict(monitor._raw_metrics).to_dict('list')
        keys = monitor.minimize + monitor.maximize
        pnum_ = kwplot.PlotNums(nSubplots=len(keys))
        for i, key in enumerate(keys):
            kwplot.multi_plot(
                monitor._epochs, {'raw ' + key: raw_ydatas[key],
                                  'smooth ' + key: smooth_ydatas[key]},
                xlabel='epoch', ylabel=key, pnum=pnum_[i], fnum=1,
                # markers={'raw ' + key: '-', 'smooth ' + key: '--'},
                # colors={'raw ' + key: 'b', 'smooth ' + key: 'b'},
            )

            # star all the good epochs
            monitor.best_epochs(1)
            flags = np.array(monitor._is_good)
            if np.any(flags):
                plt.plot(list(ub.compress(monitor._epochs, flags)),
                         list(ub.compress(smooth_ydatas[key], flags)), 'b*')

    def __getstate__(monitor):
        state = monitor.__dict__.copy()
        _ewma = state.pop('_ewma')
        state['ewma_state'] = _ewma.__dict__
        return state

    def __setstate__(monitor, state):
        ewma_state = state.pop('ewma_state', None)
        if ewma_state is not None:
            monitor._ewma = util.ExpMovingAve()
            monitor._ewma.__dict__.update(ewma_state)
        monitor.__dict__.update(**state)

    def state_dict(monitor):
        """
        pytorch-like API. Alias for __getstate__
        """
        return monitor.__getstate__()

    def load_state_dict(monitor, state):
        """
        pytorch-like API. Alias for __setstate__

        Args:
            state (Dict):
        """
        return monitor.__setstate__(state)

    def update(monitor, epoch, _raw_metrics, lr=None):
        """
        Informs the monitor about quality measurements for a particular epoch.

        Args:
            epoch (int):
                Current epoch number

            _raw_metrics (Dict[str, float]):
                Scalar values for each quality metric that was measured on this
                epoch.

        Returns:
            bool: improved:
                True if the model has quality of the validation metrics have
                improved.

        Example:
            >>> from .monitor import *  # NOQA
            >>> from viame.arrows.pytorch.netharn import core as nh
            >>> rng = np.random.RandomState(0)
            >>> monitor = Monitor(minimize=['loss'], min_lr=1e-5)
            >>> for epoch in range(200):
            >>>     _raw_metrics = {'loss': rng.rand(), 'miou': rng.rand()}
            >>>     lr = 1.0 / (10 ** max(1, epoch))
            >>>     monitor.update(epoch, _raw_metrics, lr)
            >>>     term_reason = monitor.is_done()
            >>>     if term_reason:
            >>>         print('MONITOR IS DONE. BREAK')
            >>>         break
            >>> print(monitor.is_done())
            >>> assert monitor._current_lr <= monitor.min_lr
            >>> print('monitor = {!r}'.format(monitor))
        """
        monitor._epochs.append(epoch)
        monitor._raw_metrics.append(_raw_metrics)
        monitor._ewma.update(_raw_metrics)
        # monitor.other_data.append(other)

        if epoch is not None:
            monitor._current_epoch = epoch

        if lr is not None:
            monitor._current_lr = lr

        _smooth_metrics = monitor._ewma.average()
        monitor._smooth_metrics.append(_smooth_metrics.copy())

        improved_keys = monitor._improved(_smooth_metrics, monitor._best_smooth_metrics)
        if improved_keys:

            ignore_this_epoch = False
            if monitor._current_epoch is not None:
                # If we are ignoring the monitor in the first few epochs then
                # dont store the metrics
                if monitor._current_epoch < monitor.ignore_first_epochs:
                    ignore_this_epoch = True

            if not ignore_this_epoch:
                if monitor._best_smooth_metrics is None:
                    monitor._best_smooth_metrics = _smooth_metrics.copy()
                    monitor._best_raw_metrics = _raw_metrics.copy()
                else:
                    for key in improved_keys:
                        monitor._best_smooth_metrics[key] = _smooth_metrics[key]
                        monitor._best_raw_metrics[key] = _raw_metrics[key]

            monitor._best_epoch = epoch
            monitor._n_bad_epochs = 0
        else:
            monitor._n_bad_epochs += 1

        improved = len(improved_keys) > 0
        monitor._is_good.append(improved)
        return improved

    def _improved(monitor, metrics, best_metrics):
        """
        If any of the metrics we care about is improving then we are happy

        Returns:
            List[str]: list of the quality metrics that have improved

        Example:
            >>> from .monitor import *
            >>> monitor = Monitor(['loss'], ['acc'])
            >>> metrics = {'loss': 5, 'acc': .99}
            >>> best_metrics = {'loss': 4, 'acc': .98}
            >>> monitor._improved(metrics, best_metrics)
            ['acc']
        """
        keys = monitor.maximize + monitor.minimize

        def _as_minimization(metrics):
            # convert to a minimization problem
            sign = np.array(([-1] * len(monitor.maximize)) +
                            ([1] * len(monitor.minimize)))
            chosen = np.array(list(ub.take(metrics, keys)))
            return chosen, sign

        current, sign1 = _as_minimization(metrics)

        if not best_metrics:
            return keys

        best, sign2 = _as_minimization(best_metrics)

        # TODO: also need to see if anything got significantly worse

        # only use threshold rel mode
        monitor.rel_threshold = 1e-6
        rel_epsilon = 1.0 - monitor.rel_threshold
        improved_flags = (sign1 * current) < (rel_epsilon * sign2 * best)

        improved_keys = list(ub.compress(keys, improved_flags))
        return improved_keys

    def is_done(monitor):
        """
        Returns True if the termination criterion is satisfied

        Returns:
            bool | str: False if training should continue, otherwise returns a
                string indicating the reason training should be stopped.

        Example:
            >>> from .monitor import *
            >>> Monitor().is_done()
            False
            >>> print(Monitor(patience=0).is_done())
            Validation set is not improving, terminating ...
        """
        if monitor.max_epoch is not None and monitor._current_epoch is not None:
            if monitor._current_epoch >= monitor.max_epoch:
                return 'Maximum harn.epoch reached, terminating ...'
        if monitor.min_lr is not None and monitor._current_lr is not None:
            if monitor._current_lr <= monitor.min_lr:
                return 'Minimum lr reached, terminating ...'
        if monitor.patience is not None and monitor._n_bad_epochs is not None:
            if monitor._n_bad_epochs >= monitor.patience:
                return 'Validation set is not improving, terminating ...'
        return False

    def message(monitor, ansi=True):
        """
        A status message with optional ANSI coloration

        Args:
            ansi (bool, default=True): if False disables ANSI coloration

        Returns:
            str: message for logging

        Example:
            >>> from .monitor import *
            >>> monitor = Monitor(smoothing=0.6)
            >>> print(monitor.message(ansi=False))
            vloss is unevaluated
            >>> monitor.update(0, {'loss': 1.0})
            >>> print(monitor.message(ansi=False))
            vloss: 1.0000 (n_bad=00, best=1.0000)
            >>> monitor.update(0, {'loss': 2.0})
            >>> print(monitor.message(ansi=False))
            vloss: 1.4000 (n_bad=01, best=1.0000)
            >>> monitor.update(0, {'loss': 0.1})
            >>> print(monitor.message(ansi=False))
            vloss: 0.8800 (n_bad=00, best=0.8800)

        Example:
            >>> # Test case for ignore_first_epochs
            >>> monitor = Monitor(smoothing=0.6, ignore_first_epochs=2)
            >>> monitor.update(0, {'loss': 0.1})
            >>> print(monitor.message(ansi=False))
            >>> monitor.update(1, {'loss': 1.1})
            >>> print(monitor.message(ansi=False))
            >>> monitor.update(2, {'loss': 0.3})
            >>> print(monitor.message(ansi=False))
            >>> monitor.update(3, {'loss': 0.2})
            >>> print(monitor.message(ansi=False))
            vloss: 0.1000 (n_bad=00, best=ignored)
            vloss: 0.5000 (n_bad=00, best=ignored)
            vloss: 0.4200 (n_bad=00, best=0.4200)
            vloss: 0.3320 (n_bad=00, best=0.3320)

        """
        if not monitor._epochs:
            message = 'vloss is unevaluated'
            if ansi:
                message = ub.color_text(message, 'blue')
        else:
            if monitor._smooth_metrics is None:
                prev_loss_str = 'unknown'
            else:
                prev_loss = monitor._smooth_metrics[-1]['loss']
                prev_loss_str = '{:.4f}'.format(prev_loss)

            if monitor._best_smooth_metrics is None:
                best_loss_str = 'ignored'
            else:
                best_loss = monitor._best_smooth_metrics['loss']
                best_loss_str = '{:.4f}'.format(best_loss)

            message = 'vloss: {} (n_bad={:02d}, best={})'.format(
                prev_loss_str, monitor._n_bad_epochs, best_loss_str,
            )
            if monitor.patience is None:
                patience = monitor.max_epoch
            else:
                patience = monitor.patience
            if ansi:
                if monitor._n_bad_epochs <= int(patience * .25):
                    message = ub.color_text(message, 'green')
                elif monitor._n_bad_epochs >= int(patience * .75):
                    message = ub.color_text(message, 'red')
                else:
                    message = ub.color_text(message, 'yellow')
        return message

    def best_epochs(monitor, num=None, smooth=True):
        """
        Returns the best `num` epochs for every metric.

        Args:
            num (int, default=None):
                Number of top epochs to return. If not specified then all are
                returned.

            smooth (bool, default=True):
                Uses smoothed metrics if True otherwise uses the raw metrics.

        Returns:
            Dict[str, ndarray]: epoch numbers for all of the best epochs

        Example:
            >>> monitor = demodata_monitor()
            >>> metric_ranks = monitor.best_epochs(5)
            >>> print(ub.repr2(metric_ranks, with_dtype=False, nl=1))
            {
                'loss': np.array([297, 296, 299, 295, 298]),
                'miou': np.array([299, 296, 298, 295, 292]),
            }
        """
        metric_ranks = {}
        for key in it.chain(monitor.minimize, monitor.maximize):
            metric_ranks[key] = monitor._rank(key, smooth=smooth)[:num]
        return metric_ranks

    def _rank(monitor, key, smooth=True):
        """
        Ranks the best epochs from best to worst for each metric

        Example:
            >>> monitor = demodata_monitor()
            >>> ranked_epochs = monitor._rank('loss', smooth=False)
            >>> ranked_epochs = monitor._rank('miou', smooth=True)

            >>> monitor = demodata_monitor(ignore_first_epochs=10)
            >>> ranked_epochs = monitor._rank('loss', smooth=False)
            >>> assert 1 not in ranked_epochs
            >>> ranked_epochs = monitor._rank('miou', smooth=True)
            >>> assert 1 not in ranked_epochs
        """
        if smooth:
            metrics = monitor._smooth_metrics
        else:
            metrics = monitor._raw_metrics

        epochs = np.array(monitor._epochs)
        values = np.array([m[key] for m in metrics])
        is_valid = np.array(
            [False if e is None else
             int(e) >= int(monitor.ignore_first_epochs)
             for e in monitor._epochs], dtype=bool)

        valid_values = values[is_valid]
        valid_epochs = epochs[is_valid]

        if key in monitor.maximize:
            valid_sortx = np.argsort(valid_values)[::-1]
        elif key in monitor.minimize:
            valid_sortx = np.argsort(valid_values)
        else:
            raise KeyError(type)
        ranked_epochs = valid_epochs[valid_sortx]
        return ranked_epochs

    def _BROKEN_rank_epochs(monitor):
        """
        FIXME:
            broken - implement better rank aggregation with custom weights

        Example:
            >>> monitor = demodata_monitor()
            >>> monitor._BROKEN_rank_epochs()
        """
        rankings = {}
        for key, value in monitor.best_epochs(smooth=False).items():
            rankings[key + '_raw'] = value

        for key, value in monitor.best_epochs(smooth=True).items():
            rankings[key + '_smooth'] = value

        # borda-like weighted rank aggregation.
        # probably could do something better.
        epoch_to_weight = ub.ddict(lambda: 0)
        for key, ranking in rankings.items():
            # weights = np.linspace(0, 1, num=len(ranking))[::-1]
            weights = np.logspace(0, 2, num=len(ranking))[::-1] / 100
            for epoch, w in zip(ranking, weights):
                epoch_to_weight[epoch] += w

        agg_ranking = ub.argsort(epoch_to_weight)[::-1]
        return agg_ranking

if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m netharn.monitor all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
