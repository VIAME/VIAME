"""
Under development! Function names and logic may change at any time. Nothing in
this file should be considered as stable! Use at your own risk.

These are methods that you can mixin to your FitHarn implementation to extend
its functionality to typical, but non-default cases.

The purpose of this file is to contain functions that might not general-purpose
enough to add to FitHarn itself, but they are also common enough, where it
makes no sense to write them from scratch for each new project.
"""
try:  # nocover
    from packaging.version import parse as LooseVersion
except ImportError:
    from distutils.version import LooseVersion


def _dump_monitor_tensorboard(harn, mode='epoch', special_groupers=['loss'],
                              serial=False):
    """
    Dumps PNGs to disk visualizing tensorboard scalars.
    Also dumps pickles to disk containing the same information.

    Args:
        mode (str | Tuple[str], default='epoch'):
            Can be either `epoch` or `iter`, or a tuple containing both.

        special_groupers (List[str], default=['loss']):
            list of strings indicating groups.  For each item, a logged value
            is contained in that group if it contains that item as a substring.

        serial (bool, default=False):
            If True executes the drawing process in the main process, otherwise
            it forks a new process and runs in the background.

    CommandLine:
        xdoctest -m netharn.mixins _dump_monitor_tensorboard --profile

    Example:
        >>> from viame.arrows.pytorch.netharn import core as nh
        >>> from .mixins import _dump_monitor_tensorboard
        >>> harn = nh.FitHarn.demo()
        >>> harn.run()
        >>> try:
        >>>     _dump_monitor_tensorboard(harn)
        >>> except ImportError:
        >>>     pass
    """
    import ubelt as ub
    from viame.arrows.pytorch.netharn import core as nh
    from os.path import join
    import json
    import six
    from six.moves import cPickle as pickle

    # harn.debug('Plotting tensorboard data. serial={}, mode={}'.format(serial, mode))

    train_dpath = harn.train_dpath

    tb_data = nh.util.read_tensorboard_scalars(train_dpath, cache=0, verbose=0)

    tb_data['meta'] = {
        'nice': harn.hyper.nice,
        'special_groupers': special_groupers,
    }

    out_dpath = ub.ensuredir((train_dpath, 'monitor', 'tensorboard'))

    # Write a script that the user can run to
    if not ub.WIN32:
        reviz_fpath = join(out_dpath, 'revisualize.sh')
        reviz_text = ub.codeblock(
            '''
            #!/bin/bash
            __heredoc__ = """
            Helper script to visualize all of the results in the pkl / json files
            in this directory.
            """
            REVIZ_DPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
            xdoctest -m netharn.mixins _dump_measures --out_dpath=$REVIZ_DPATH
            ''')
        with open(reviz_fpath, 'w') as file:
            file.write(reviz_text)
        try:
            import os
            import stat
            orig_mode = os.stat(reviz_fpath).st_mode
            new_flags = stat.S_IXGRP | stat.S_IEXEC
            if (new_flags & orig_mode) != new_flags:
                new_mode = orig_mode | new_flags
                os.chmod(reviz_fpath, new_mode)
        except Exception as ex:
            print('ex = {!r}'.format(ex))

    tb_data_pickle_fpath = join(out_dpath, 'tb_data.pkl')
    with open(tb_data_pickle_fpath, 'wb') as file:
        pickle.dump(tb_data, file)

    tb_data_json_fpath = join(out_dpath, 'tb_data.json')
    with open(tb_data_json_fpath, 'w') as file:
        if six.PY2:
            jsonkw = dict(indent=1)
        else:
            jsonkw = dict(indent=' ')
        try:
            json.dump(tb_data, file, **jsonkw)
        except Exception as ex:
            print('ex = {!r}'.format(ex))
            json.dump({
                'error': 'Unable to write to json.',
                'info': 'See pickle file: {}'.format(tb_data_json_fpath)},
                file, **jsonkw)

    # The following function draws the tensorboard result
    # This might take a some non-trivial amount of time so we attempt to run in
    # a separate process.
    func = _dump_measures
    args = (tb_data, out_dpath, mode)

    if not serial:

        if False:
            # Maybe thread-safer way of doing this? Maybe not, there is a
            # management thread used by futures.
            from concurrent import futures
            if not hasattr(harn, '_internal_executor'):
                harn._internal_executor = futures.ProcessPoolExecutor(max_workers=1)
                harn._prev_job = None
            if harn._prev_job is None or harn._prev_job.done():
                # Wait to before submitting another job
                # Unsure if its ok that this job might not be a daemon
                harn.info('DO MPL DRAW')
                job = harn._internal_executor.submit(func, *args)
                harn._prev_job = job
            else:
                if harn._prev_job is not None:
                    harn.info('NOT DOING MPL DRAW')
                    harn.warn('NOT DOING MPL DRAW')
        else:
            # This causes thread-unsafe warning messages in the inner loop
            # Likely because we are forking while a thread is alive
            if not hasattr(harn, '_internal_procs'):
                harn._internal_procs = ub.ddict(dict)

            # Clear finished processes from the pool
            for pid in list(harn._internal_procs[mode].keys()):
                proc = harn._internal_procs[mode][pid]
                if not proc.is_alive():
                    harn._internal_procs[mode].pop(pid)

            # only start a new process if there is room in the pool
            if len(harn._internal_procs[mode]) < 1:
                import multiprocessing
                proc = multiprocessing.Process(target=func, args=args)
                proc.daemon = True
                proc.start()
                harn._internal_procs[mode][proc.pid] = proc
            else:
                if 0:
                    harn.warn('NOT DOING MPL DRAW')
    else:
        func(*args)


def _redump_measures(dpath):
    """
    """
    import json
    from os.path import join

    import kwplot
    kwplot.autompl(force='agg')

    try:
        import seaborn as sns
        sns.set()
    except ImportError:
        pass

    fpath = join(dpath, 'tb_data.json')
    tb_data = json.load(open(fpath, 'r'))

    out_dpath = dpath
    mode = 'epoch'
    _dump_measures(tb_data, out_dpath, mode)


def _dump_measures(tb_data, out_dpath, mode=None, smoothing=0.0,
                   ignore_outliers=True):
    """
    This is its own function in case we need to modify formatting

    CommandLine:
        xdoctest -m netharn.mixins _dump_measures --out_dpath=.

    Example:
        >>> # SCRIPT
        >>> # Reread a dumped pickle file
        >>> from .mixins import *  # NOQA
        >>> from .mixins import _dump_monitor_tensorboard, _dump_measures
        >>> import json
        >>> from os.path import join
        >>> import ubelt as ub
        >>> try:
        >>>     import seaborn as sns
        >>>     sns.set()
        >>> except ImportError:
        >>>     pass
        >>> out_dpath = ub.expandpath('~/work/project/fit/nice/nicename/monitor/tensorboard/')
        >>> out_dpath = ub.argval('--out_dpath', default=out_dpath)
        >>> mode = ['epoch', 'iter']
        >>> fpath = join(out_dpath, 'tb_data.json')
        >>> tb_data = json.load(open(fpath, 'r'))
        >>> import kwplot
        >>> kwplot.autompl()
        >>> _dump_measures(tb_data,  out_dpath, smoothing=0)
    """
    import ubelt as ub
    from os.path import join
    import numpy as np
    import kwplot
    import matplotlib as mpl
    from kwplot.auto_backends import BackendContext

    with BackendContext('agg'):
        # kwplot.autompl()

        # TODO: Is it possible to get htop to show this process with some name that
        # distinguishes it from the dataloader workers?
        # import sys
        # import multiprocessing
        # if multiprocessing.current_process().name != 'MainProcess':
        #     if sys.platform.startswith('linux'):
        #         import ctypes
        #         libc = ctypes.cdll.LoadLibrary('libc.so.6')
        #         title = 'Netharn MPL Dump Measures'
        #         libc.prctl(len(title), title, 0, 0, 0)

        # NOTE: This cause warnings when exeucted as daemon process
        # try:
        #     import seaborn as sbn
        #     sbn.set()
        # except ImportError:
        #     pass

        valid_modes = ['epoch', 'iter']
        if mode is None:
            mode = valid_modes
        if ub.iterable(mode):
            # Hack: Call with all modes
            for mode_ in mode:
                _dump_measures(tb_data, out_dpath, mode=mode_, smoothing=smoothing,
                               ignore_outliers=ignore_outliers)
            return
        else:
            assert mode in valid_modes

        meta = tb_data.get('meta', {})
        nice = meta.get('nice', '?nice?')
        special_groupers = meta.get('special_groupers', ['loss'])

        fig = kwplot.figure(fnum=1)

        plot_keys = [key for key in tb_data if
                     ('train_' + mode in key or
                      'vali_' + mode in key or
                      'test_' + mode in key or
                      mode + '_' in key)]
        y01_measures = [
            '_acc', '_ap', '_mAP', '_auc', '_mcc', '_brier', '_mauc',
        ]
        y0_measures = ['error', 'loss']

        keys = set(tb_data.keys()).intersection(set(plot_keys))

        # print('mode = {!r}'.format(mode))
        # print('tb_data.keys() = {!r}'.format(tb_data.keys()))
        # print('plot_keys = {!r}'.format(plot_keys))
        # print('keys = {!r}'.format(keys))

        def smooth_curve(ydata, beta):
            """
            Curve smoothing algorithm used by tensorboard
            """
            import pandas as pd
            alpha = 1.0 - beta
            if alpha <= 0:
                return ydata
            ydata_smooth = pd.Series(ydata).ewm(alpha=alpha).mean().values
            return ydata_smooth

        def inlier_ylim(ydatas):
            """
            outlier removal used by tensorboard
            """
            low, high = None, None
            for ydata in ydatas:
                q1 = 0.05
                q2 = 0.95
                low_, high_ = np.quantile(ydata, [q1, q2])

                # Extrapolate how big the entire span should be based on inliers
                inner_q = q2 - q1
                inner_extent = high_ - low_
                extrap_total_extent = inner_extent  / inner_q

                # amount of padding to add to either side
                missing_p1 = q1
                missing_p2 = 1 - q2
                frac1 = missing_p1 / (missing_p2 + missing_p1)
                frac2 = missing_p2 / (missing_p2 + missing_p1)
                missing_extent = extrap_total_extent - inner_extent

                pad1 = missing_extent * frac1
                pad2 = missing_extent * frac2

                low_ = low_ - pad1
                high_ = high_ + pad2

                low = low_ if low is None else min(low_, low)
                high = high_ if high is None else max(high_, high)
            return (low, high)

        # Hack values that we don't apply smoothing to
        HACK_NO_SMOOTH = ['lr', 'momentum']

        def tag_grouper(k):
            # parts = ['train_epoch', 'vali_epoch', 'test_epoch']
            # parts = [p.replace('epoch', 'mode') for p in parts]
            parts = [p + mode for p in ['train_', 'vali_', 'test_']]
            for p in parts:
                if p in k:
                    return p.split('_')[0]
            return 'unknown'

        GROUP_LOSSES = True
        GROUP_AND_INDIVIDUAL = False
        INDIVIDUAL_PLOTS = True
        GROUP_SPECIAL = True

        if GROUP_LOSSES:
            # Group all losses in one plot for comparison
            loss_keys = [k for k in keys if 'loss' in k]
            tagged_losses = ub.group_items(loss_keys, tag_grouper)
            tagged_losses.pop('unknown', None)
            kw = {}
            kw['ymin'] = 0.0
            # print('tagged_losses = {!r}'.format(tagged_losses))
            for tag, losses in tagged_losses.items():

                min_abs_y = .01
                min_y = 0
                xydata = ub.odict()
                for key in sorted(losses):
                    ydata = tb_data[key]['ydata']

                    if HACK_NO_SMOOTH not in key.split('_'):
                        ydata = smooth_curve(ydata, smoothing)

                    try:
                        min_y = min(min_y, ydata.min())
                        pos_ys = ydata[ydata > 0]
                        min_abs_y = min(min_abs_y, pos_ys.min())
                    except Exception:
                        pass

                    xydata[key] = (tb_data[key]['xdata'], ydata)

                kw['ymin'] = min_y

                if ignore_outliers:
                    low, kw['ymax'] = inlier_ylim([t[1] for t in xydata.values()])

                yscales = ['symlog', 'linear']
                for yscale in yscales:
                    fig.clf()
                    ax = fig.gca()
                    title = nice + '\n' + tag + '_' + mode + ' losses'
                    kwplot.multi_plot(xydata=xydata, ylabel='loss', xlabel=mode,
                                      yscale=yscale, title=title, fnum=1, ax=ax,
                                      **kw)
                    if yscale == 'symlog':
                        if LooseVersion(mpl.__version__) >= LooseVersion('3.3'):
                            ax.set_yscale('symlog', linthresh=min_abs_y)
                        else:
                            ax.set_yscale('symlog', linthreshy=min_abs_y)
                    fname = '_'.join([tag, mode, 'multiloss', yscale]) + '.png'
                    fpath = join(out_dpath, fname)
                    ax.figure.savefig(fpath)

            # don't dump losses individually if we dump them in a group
            if not GROUP_AND_INDIVIDUAL:
                keys.difference_update(set(loss_keys))
                # print('keys = {!r}'.format(keys))

        if GROUP_SPECIAL:
            tag_groups = ub.group_items(keys, tag_grouper)
            tag_groups.pop('unknown', None)
            # Group items matching these strings
            kw = {}
            for tag, tag_keys in tag_groups.items():
                for groupname in special_groupers:
                    group_keys = [k for k in tag_keys if groupname in k.split('_')]
                    if len(group_keys) > 1:
                        # Gather data for this group
                        xydata = ub.odict()
                        for key in sorted(group_keys):
                            ydata = tb_data[key]['ydata']
                            if HACK_NO_SMOOTH not in key.split('_'):
                                ydata = smooth_curve(ydata, smoothing)
                            xydata[key] = (tb_data[key]['xdata'], ydata)

                        if ignore_outliers:
                            low, kw['ymax'] = inlier_ylim([t[1] for t in xydata.values()])

                        yscales = ['linear']
                        for yscale in yscales:
                            fig.clf()
                            ax = fig.gca()
                            title = nice + '\n' + tag + '_' + mode + ' ' + groupname
                            kwplot.multi_plot(xydata=xydata, ylabel=groupname, xlabel=mode,
                                              yscale=yscale, title=title, fnum=1, ax=ax,
                                              **kw)
                            if yscale == 'symlog':
                                ax.set_yscale('symlog', linthreshy=min_abs_y)
                            fname = '_'.join([tag, mode, 'group-' + groupname, yscale]) + '.png'
                            fpath = join(out_dpath, fname)
                            ax.figure.savefig(fpath)

                        if not GROUP_AND_INDIVIDUAL:
                            keys.difference_update(set(group_keys))

        if INDIVIDUAL_PLOTS:
            # print('keys = {!r}'.format(keys))
            for key in keys:
                d = tb_data[key]

                ydata = d['ydata']
                ydata = smooth_curve(ydata, smoothing)

                kw = {}
                if any(m.lower() in key.lower() for m in y01_measures):
                    kw['ymin'] = 0.0
                    kw['ymax'] = 1.0
                elif any(m.lower() in key.lower() for m in y0_measures):
                    kw['ymin'] = min(0.0, ydata.min())
                    if ignore_outliers:
                        low, kw['ymax'] = inlier_ylim([ydata])

                # NOTE: this is actually pretty slow
                fig.clf()
                ax = fig.gca()
                title = nice + '\n' + key
                kwplot.multi_plot(d['xdata'], ydata, ylabel=key, xlabel=mode,
                                  title=title, fnum=1, ax=ax, **kw)

                # png is slightly smaller than jpg for this kind of plot
                fpath = join(out_dpath, key + '.png')
                # print('save fpath = {!r}'.format(fpath))
                ax.figure.savefig(fpath)
