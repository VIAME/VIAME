# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub


def read_tensorboard_scalars(train_dpath, verbose=1, cache=1):
    """
    Reads all tensorboard scalar events in a directory.
    Caches them becuase reading events of interest from protobuf can be slow.
    """
    import glob
    from os.path import join
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        raise ImportError('tensorboard/tensorflow is not installed')
    event_paths = sorted(glob.glob(join(train_dpath, 'events.out.tfevents*')))
    # make a hash so we will re-read of we need to
    cfgstr = ub.hash_data(list(map(ub.hash_file, event_paths))) if cache else ''
    cacher = ub.Cacher('tb_scalars', depends=cfgstr, enabled=cache,
                       dpath=join(train_dpath, '_cache'))
    datas = cacher.tryload()
    if datas is None:
        datas = {}
        for p in ub.ProgIter(list(reversed(event_paths)), desc='read tensorboard', enabled=verbose):
            ea = event_accumulator.EventAccumulator(p)
            ea.Reload()
            for key in ea.scalars.Keys():
                if key not in datas:
                    datas[key] = {'xdata': [], 'ydata': [], 'wall': []}
                subdatas = datas[key]
                events = ea.scalars.Items(key)
                for e in events:
                    subdatas['xdata'].append(int(e.step))
                    subdatas['ydata'].append(float(e.value))
                    subdatas['wall'].append(float(e.wall_time))

        # Order all information by its wall time
        for key, subdatas in datas.items():
            sortx = ub.argsort(subdatas['wall'])
            for d, vals in subdatas.items():
                subdatas[d] = list(ub.take(vals, sortx))
        cacher.save(datas)
    return datas
