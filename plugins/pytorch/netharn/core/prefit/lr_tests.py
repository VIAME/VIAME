import torch  # NOQA
import copy
import ubelt as ub
import numpy as np


class TestResult(ub.NiceRepr):
    def __init__(self, a, b):
        self.records = a
        self.recommended_lr = b

    def __nice__(self):
        return 'recommended_lr = {!r}'.format(self.recommended_lr)


def lr_range_test(harn, init_value=1e-8, final_value=10., beta=0.98,
                  explode_factor=10, num_iters=100):
    """
    Implementation of Leslie Smith's LR-range described in [2] test based on
    code found in [1].

    Args:
        init_value : initial learning rate
        beta (float): smoothing param

    Notes:
        It is critical that `init_value` starts off much lower than the actual
        valid LR-range. This is because this test actually modifies a copy of
        the model parameters as it runs, so the metrics (i.e. loss) of each LR
        actually benefits from all steps taken in previous iterations with
        previous lr values.

    References:
        .. _[1]: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
        .. _[2]: https://arxiv.org/abs/1803.09820
        .. _[3]: https://github.com/fastai/fastai/blob/e6b56de53f80d2b2d39037c82d3a23ce72507cd7/old/fastai/sgdr.py

    Example:
        >>> from .prefit.lr_tests import *
        >>> from viame.arrows.pytorch.netharn import core as nh
        >>> harn = nh.FitHarn.demo().initialize()
        >>> result = lr_range_test(harn)
        >>> print('result = {!r}'.format(result))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> result.draw()
        >>> kwplot.show_if_requested()

    Ignore:
        >>> import sys
        >>> sys.path.append('/home/joncrall/code/netharn/examples')
        >>> from mnist import setup_harn
        >>> harn = setup_harn().initialize()
        >>> harn.preferences['prog_backend'] = 'progiter'
        >>> result = lr_range_test(harn)
        >>> print('result = {!r}'.format(result))
        >>> result.draw()

    Ignore:
        >>> import sys
        >>> sys.path.append('/home/joncrall/code/netharn/examples')
        >>> from cifar import setup_harn
        >>> from mnist import setup_harn
        >>> harn = setup_harn().initialize()

        from .mixins import *
        import xdev
        globals().update(xdev.get_func_kwargs(lr_range_test))

        init_value = 1e-5
        final_value = 0.1
    """
    from viame.arrows.pytorch.netharn import core as nh

    # Save the original state
    orig_model_state = copy.deepcopy(harn.model.state_dict())
    orig_optim_state = copy.deepcopy(harn.optimizer.state_dict())
    _orig_on_batch = harn.on_batch
    _orig_on_epoch = harn.on_epoch

    try:
        # TODO: find a way to disable callabacks nicely
        harn.on_batch = lambda *args, **kw: None
        harn.on_epoch = lambda *args, **kw: None

        def set_optimizer_property(optimizer, attr, value):
            for group in optimizer.param_groups:
                group[attr] = value

        def get_optimizer_property(optimizer, attr):
            return [group[attr] for group in optimizer.param_groups]

        optimizer = harn.optimizer
        ub.inject_method(optimizer, set_optimizer_property, 'set_property')
        ub.inject_method(optimizer, get_optimizer_property, 'get_property')

        orig_state = harn.model.state_dict()

        # This line seems like a safe way to reset model weights
        harn.model.load_state_dict(orig_state)

        tag = 'train'
        loader = harn.loaders[tag]

        num_epochs = min(num_iters, len(loader))

        # These are the learning rates we will scan through
        learning_rates = np.logspace(
            np.log10(init_value), np.log10(final_value),
            num=num_epochs, base=10, endpoint=False)

        # Bookkeeping
        metrics = nh.util.ExpMovingAve(alpha=1 - beta, correct_bias=True)
        best_loss = float('inf')
        best_lr = init_value
        records = ub.ddict(list)

        # Note: ignore the loss for thie first iteration. The loss computed here
        # corresponds to the original model without any SGD steps, so it doesn't
        # belong to any of the learning rates in our scan range.
        optimizer.set_property('lr', init_value)

        batch_iter = iter(loader)
        prog = ub.ProgIter(range(num_epochs), desc='running lr test')
        for bx in prog:
            # The LR that corresponds to the loss in the iteration belongs
            # to the LR that was set in the previous iteration.
            curr_lr = 0 if bx == 0 else optimizer.get_property('lr')[0]

            # Update the lr for the next step, but the loss computed was for
            # the previous iteration's LR value.
            next_lr = learning_rates[bx]
            optimizer.set_property('lr', next_lr)
            optimizer.zero_grad()

            # Get the loss for this mini-batch of inputs/outputs
            raw_batch = next(batch_iter)
            batch = harn.prepare_batch(raw_batch)
            outputs, loss = harn.run_batch(batch)
            if isinstance(loss, dict):
                loss = sum(loss.values())
            raw_loss = float(loss.data.cpu().item())

            # Compute the smoothed loss
            metrics.update({'loss': raw_loss})
            curr_loss = metrics.mean()['loss']

            # Record the best loss
            if curr_loss < best_loss:
                best_loss = curr_loss
                best_lr = curr_lr

            prog.set_extra(' best_lr={:.2g}, curr_lr={:.2g}, best_loss={:.2f}, curr_loss={:.2f}'.format(best_lr, curr_lr, best_loss, curr_loss))

            if bx > 0:
                # This loss was achived by a step with the previous lr, so
                # ensure we are associating the correct lr with the loss that
                # corresponds to it.
                records['loss_std'].append(metrics.std()['loss'])
                records['loss'].append(curr_loss)
                records['raw_loss'].append(raw_loss)
                records['lr'].append(curr_lr)

                # Stop if the loss is exploding
                if curr_loss > explode_factor * best_loss:
                    prog.update()
                    prog.ensure_newline()
                    print('\nstopping because loss is exploding')
                    break

            # Do SGD step, so now the nest loss computation correspond to
            # the first learning rate in our scan range.
            loss.backward()
            optimizer.step()

        # TODO: ensure that the loader is cleaned up properly
        prog.end()
        prog = None
        if hasattr(batch_iter, 'shutdown'):
            batch_iter._shutdown_workers()
        batch_iter = None
        loader = None

        best_x = ub.argmin(records['loss'])
        best_lr = records['lr'][best_x]
        print('best_lr = {!r}'.format(best_lr))

        recommended_lr = best_lr / 10
        # recommended_lr = best_lr
        print('recommended_lr = {!r}'.format(recommended_lr))
    except Exception:
        raise
    finally:
        # Reset model back to its original state
        harn.optimizer.load_state_dict(orig_optim_state)
        harn.model.load_state_dict(orig_model_state)
        harn.on_batch = _orig_on_batch
        harn.on_epoch = _orig_on_epoch

    def draw():
        import kwplot
        kwplot.autompl()
        ylimits = records['loss'] + (6 * records['loss_std'])
        ymax = np.percentile(ylimits, 96.9) / .9
        kwplot.multi_plot(
            xdata=records['lr'],
            ydata=records['loss'],
            spread=records['loss_std'],
            xlabel='learning-rate',
            ylabel='smoothed-loss',
            xscale='log',
            ymin=0,
            ymax=ymax,
            xmin='data',
            xmax='data',
            # xmin=min(records['lr']),
            # xmax=max(records['lr']),
            doclf=True,
            fnum=1,
        )

    result = TestResult(records, recommended_lr)
    result.draw = draw
    return result


def lr_range_scan(harn, low=1e-6, high=10.0, num=8, niter_train=1,
                  niter_vali=0, scale='log'):
    """
    This takes longer than lr_range_test, but is theoretically more robust.

    Args:
        low (float, default=1e-6): minimum lr to scan
        high (float, default=1.0): maximum lr to scan
        num (int, default=32): number of lrs to scan
        niter_train (int, default=10): number of training batches to test for each lr
        niter_vali (int, default=0): number of validation batches to test for each lr

    Notes:
        New better lr test:
            * For each lr in your scan range
            * reset model to original state
            * run a few batches and backprop after each
            * iterate over those batches again and compute their
              loss, but dont backprop this time.
            * associate the average of those losses with the learning rate.

    Example:
        >>> from .prefit.lr_tests import *
        >>> from viame.arrows.pytorch.netharn import core as nh
        >>> harn = nh.FitHarn.demo().initialize()
        >>> result = lr_range_scan(harn)
        >>> print('result = {!r}'.format(result))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> result.draw()
        >>> kwplot.show_if_requested()

    Ignore:
        >>> import sys
        >>> sys.path.append('/home/joncrall/code/netharn/examples')
        >>> from mnist import setup_harn
        >>> harn = setup_harn().initialize()
        >>> harn.preferences['prog_backend'] = 'progiter'
        >>> result = lr_range_scan(harn, niter_train=100, niter_vali=10, num=32)
        >>> print('result = {!r}'.format(result))
        >>> result.draw()

    Ignore:
        >>> from .mixins import *
        >>> import sys
        >>> sys.path.append('/home/joncrall/code/netharn/examples')
        >>> from ggr_matching import setup_harn
        >>> harn = setup_harn(workers=6, xpu=0).initialize()
        >>> result = lr_range_scan(harn, niter_train=6 * 2)
        >>> print('recommended_lr = {!r}'.format(recommended_lr))
        >>> draw()

    TODO:
        - [ ] ensure that this is a randomized sample of the validation
        dataset.
        - [ ] cache the dataset if it fits into memory after we run the first
        epoch.
    """
    from viame.arrows.pytorch.netharn import core as nh
    use_vali = bool(niter_vali)

    # These are the learning rates we will scan through
    if scale == 'linear':
        learning_rates = np.linspace(low, high, num=num,
                                     endpoint=True)
    elif scale == 'log':
        log_b, base = np.log10, 10
        learning_rates = np.logspace(log_b(low), log_b(high),
                                     num=num, base=base, endpoint=True)
    else:
        raise KeyError(scale)

    orig_model_state = copy.deepcopy(harn.model.state_dict())
    orig_optim_state = copy.deepcopy(harn.optimizer.state_dict())
    _orig_on_batch = harn.on_batch
    _orig_on_epoch = harn.on_epoch

    try:
        failed_lr = None
        best_lr = 0
        best_loss = float('inf')

        records = {}
        records['train'] = ub.ddict(list)
        if use_vali:
            records['vali'] = ub.ddict(list)

        # TODO: find a way to disable callabacks nicely
        harn.on_batch = lambda *args, **kw: None
        harn.on_epoch = lambda *args, **kw: None

        prog = harn._make_prog(learning_rates, desc='scan learning rates',
                               disable=not harn.preferences['show_prog'],
                               total=len(learning_rates), leave=True,
                               dynamic_ncols=True, position=0)

        harn.info('Running lr-scan')
        for lr in prog:
            # prog.set_description(ub.color_text('scan lr = {:.2g}'.format(lr), 'darkgreen'))
            if hasattr(prog, 'ensure_newline'):
                prog.ensure_newline()
            # Reset model back to its original state
            harn.optimizer.load_state_dict(orig_optim_state)
            harn.model.load_state_dict(orig_model_state)

            # Set the optimizer to the current lr we are scanning
            harn.optimizer.param_groups[0]['lr'] = lr

            # Run a partial training and validation epoch
            try:
                epoch_metrics = {}
                epoch_metrics['train'] = harn._demo_epoch('train', learn=True, max_iter=niter_train)
                curr_loss = epoch_metrics['train']['loss']

                if curr_loss > 1000:

                    raise nh.fit_harn.TrainingDiverged

                if use_vali:
                    epoch_metrics['vali'] = harn._demo_epoch('vali', learn=False, max_iter=niter_vali)
                    curr_loss = epoch_metrics['vali']['loss']

                if curr_loss < best_loss:
                    best_lr = lr
                    best_loss = curr_loss
                    curr_text = ub.color_text('curr_lr={:.2g}, best_loss={:.2f}'.format(lr, curr_loss), 'green')
                else:
                    curr_text = ub.color_text('curr_lr={:.2g}, best_loss={:.2f}'.format(lr, curr_loss), 'red')

                if hasattr(prog, 'ensure_newline'):
                    prog.set_extra(
                        curr_text + ', ' +
                        ub.color_text('best_lr={:.2g}, best_loss={:.2f}'.format(best_lr, best_loss), 'white')
                    )
                    prog.update()
                    prog.ensure_newline()
            except nh.fit_harn.TrainingDiverged:
                harn.info('Learning is causing divergence, stopping lr-scan')
                for tag in records.keys():
                    records[tag]['lr'].append(lr)
                    records[tag]['loss'].append(1000)
                raise

            for tag, epoch_metrics in epoch_metrics.items():
                records[tag]['lr'].append(lr)
                for key, value in epoch_metrics.items():
                    records[tag][key].append(value)

        harn.info('Finished lr-scan')

    except nh.fit_harn.TrainingDiverged:
        failed_lr = lr
        print('failed_lr = {!r}'.format(failed_lr))
    except Exception:
        failed_lr = lr
        print('failed_lr = {!r}'.format(failed_lr))
        raise
    finally:
        # Reset model back to its original state
        harn.optimizer.load_state_dict(orig_optim_state)
        harn.model.load_state_dict(orig_model_state)
        harn.on_batch = _orig_on_batch
        harn.on_epoch = _orig_on_epoch

    # Choose an lr to recommend
    tag = 'vali' if 'vali' in records else 'train'
    if records.get(tag, []):
        best_idx = ub.argmin(records[tag]['loss'])
        best_lr = records[tag]['lr'][best_idx]
        # Because we aren't doing crazy EWMAs and because we reset weights, we
        # should be able to simply recommend the lr that minimized the loss.
        # We may be able to do parabolic extrema finding to get an even better
        # estimate in most cases.
        recommended_lr = best_lr
    else:
        recommended_lr = None

    # Give the user a way of visualizing what we did
    def draw():
        import kwplot
        plt = kwplot.autoplt()
        plotkw = dict(
            xlabel='learning-rate', ylabel='loss', xscale=scale,
            ymin=0, xmin='data', xmax=high, fnum=1,
        )
        R = 2 if 'vali' in records else 1
        for i, tag in enumerate(['train', 'vali']):
            if tag in records:
                kwplot.multi_plot(
                    xdata=records[tag]['lr'], ydata=records[tag]['loss'],
                    ymax=1.2 * np.percentile(records[tag]['loss'], 60) / .6,
                    pnum=(1, R, i), title=tag + ' lr-scan', **plotkw)
                plt.plot(best_lr, best_loss, '*')

    result = TestResult(records, recommended_lr)
    result.draw = draw
    return result
