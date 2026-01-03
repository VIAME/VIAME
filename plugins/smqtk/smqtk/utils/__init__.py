"""
SMQTK Utils - Minimal port for VIAME search functionality.
"""
import abc
import copy
import inspect
import logging
import os
import time

import six


def ncr(n, r):
    """
    N-choose-r, also known as nCr or the binomial coefficient.
    """
    import math
    r = min(r, n - r)
    if r == 0:
        return 1
    numer = 1
    for i in range(n, n - r, -1):
        numer *= i
    denom = math.factorial(r)
    return numer // denom


def merge_dict(a, b):
    """
    Merge dictionary b into a recursively, overwriting conflicting keys.

    :param a: Dictionary to merge into.
    :param b: Dictionary to merge from.
    :return: Dictionary a after merge.
    """
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            merge_dict(a[key], b[key])
        else:
            a[key] = b[key]
    return a


class SimpleTimer(object):
    """
    A simple context manager for timing code blocks.
    """

    def __init__(self, msg=None, logger=None):
        self.msg = msg
        self.logger = logger
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        if self.logger and self.msg:
            self.logger("%s: %f s", self.msg, self.elapsed())

    def elapsed(self):
        return self.end - self.start


def check_empty_iterable(iterable, func, exception):
    """
    Check if the iterable is empty and if so, raise the given exception.
    Otherwise, call the function with the iterable.
    """
    iterable = list(iterable)
    if not iterable:
        raise exception
    return func(iterable)


def safe_create_dir(d):
    """
    Safely create a directory if it doesn't exist.
    """
    if d and not os.path.isdir(d):
        os.makedirs(d)


def safe_file_write(path, b, tmp_dir=None):
    """
    Safely write bytes to a file by writing to a temp file first.
    """
    safe_create_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        f.write(b)


class SmqtkObject(object):
    """
    Base class for SMQTK objects providing consistent logging.
    """

    @property
    def _log(self):
        return logging.getLogger(
            '.'.join([self.__module__, self.__class__.__name__])
        )


class Configurable(SmqtkObject):
    """
    Mixin class providing configuration-related methods.
    """

    @classmethod
    def get_default_config(cls):
        """
        Generate and return a default configuration dictionary for this class.
        """
        if isinstance(cls.__init__, type(object.__init__)):
            return {}

        try:
            argspec = inspect.getfullargspec(cls.__init__)
        except AttributeError:
            argspec = inspect.getargspec(cls.__init__)

        args = argspec.args[1:]  # skip 'self'
        defaults = argspec.defaults or ()

        num_non_default = len(args) - len(defaults)
        config = {}
        for i, arg in enumerate(args):
            if i < num_non_default:
                config[arg] = None
            else:
                config[arg] = defaults[i - num_non_default]

        return config

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        """
        Instantiate a new instance of this class from a configuration dictionary.
        """
        if merge_default:
            c = cls.get_default_config()
            merge_dict(c, config_dict)
            config_dict = c

        return cls(**config_dict)

    def get_config(self):
        """
        Return a JSON-compliant configuration dictionary for this instance.
        """
        return self.get_default_config()


__all__ = [
    'ncr',
    'merge_dict',
    'SimpleTimer',
    'check_empty_iterable',
    'safe_create_dir',
    'safe_file_write',
    'SmqtkObject',
    'Configurable',
]
