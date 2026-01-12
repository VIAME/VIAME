# -*- coding: utf-8 -*-
"""
DEPRECATED
"""
import numpy as np


def read_h5arr(fpath):
    import h5py
    with h5py.File(fpath, 'r') as hf:
        return hf['arr_0'][...]


def write_h5arr(fpath, arr):
    import h5py
    with h5py.File(fpath, 'w') as hf:
        hf.create_dataset('arr_0', data=arr)


def read_arr(fpath):
    """
    Example:
        >>> import ubelt as ub
        >>> from viame.pytorch import netharn as nh
        >>> from os.path import join
        >>> dpath = ub.ensure_app_cache_dir('netharn', 'tests')
        >>> arr = np.random.rand(10)
        >>> fpath = join(dpath, 'arr.npy')
        >>> nh.util.write_arr(fpath, arr)
        >>> arr2 = nh.util.read_arr(fpath)
        >>> assert np.all(arr == arr2)
        >>> # xdoctest: +REQUIRES(module:h5py)
        >>> fpath = join(dpath, 'arr.h5')
        >>> nh.util.write_arr(fpath, arr)
        >>> arr2 = nh.util.read_arr(fpath)
        >>> assert np.all(arr == arr2)
    """
    if fpath.endswith('.npy'):
        return np.load(fpath)
    elif fpath.endswith('.h5'):
        return read_h5arr(fpath)
    else:
        raise KeyError(fpath)


def write_arr(fpath, arr):
    if fpath.endswith('.npy'):
        return np.save(fpath, arr)
    elif fpath.endswith('.h5'):
        return write_h5arr(fpath, arr)
    else:
        raise KeyError(fpath)
