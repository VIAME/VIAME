# -*- coding: utf-8 -*-
"""
A thin wrapper around python logging.
"""
from __future__ import print_function, unicode_literals, absolute_import
import sys
import six
import os
import logging


def _configure_logging():
    """
    Configures python logging to using KWIVER / SPROKIT environment variables
    """
    # Use the C++ logging level by default, but allow python to be different
    cxx_level = os.environ.get('KWIVER_DEFAULT_LOG_LEVEL', 'DEBUG')
    py_level = os.environ.get('KWIVER_PYTHON_DEFAULT_LOG_LEVEL', cxx_level)
    level = getattr(logging, py_level.upper())
    logfmt = '%(name)s:%(levelname)s: %(message)s'

    USE_COLOR = False
    try:
        import coloredlogs
    except ImportError:
        USE_COLOR = False

    if USE_COLOR:
        coloredlogs.DEFAULT_FIELD_STYLES['name']['color'] = 'blue'
        coloredlogs.DEFAULT_LEVEL_STYLES['debug']['color'] = 'cyan'
        # coloredlogs.DEFAULT_LEVEL_STYLES['warn']['color'] = 'yellow'
        coloredlogs.install(level=level, fmt=logfmt)
    else:
        logging.basicConfig(format=logfmt, level=level)


def print_exc(exc_info=None):
    """
    Prints a highly visible exception.

    Example:
        >>> try:
        >>>     raise Exception('foobar')
        >>> except Exception as ex:
        >>>     import sys
        >>>     exc_info = sys.exc_info()
        >>>     print_exc(exc_info)
    """
    import traceback
    if exc_info is None:
        exc_info = sys.exc_info()
    tbtext = ''.join(traceback.format_exception(*exc_info))

    lines = [
        '',
        '┌───────────',
        '│ EXCEPTION:',
        '',
        tbtext,
        '└───────────',
        ''
    ]
    text = '\n'.join(lines)
    print(text)


def exc_report(func):
    """
    Prints a message if an exception occurs in the decorated function.

    Modules called from C++ (e.g. pybind11) are not gaurenteed to print
    exceptions if they occur (unless the C++ program does the work). This
    decorator can be used as a workaround to ensure that there is some
    indication of when Python code raises an exception.
    """
    def _exc_report_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            print_exc()
            raise
    return _exc_report_wrapper


class SprokitLogger(object):
    """
    Lighweight wrapper around python's logging.Logger class.
    """
    @exc_report
    def __init__(self, name):
        self._logger = logging.getLogger(name)
        self.log('debug', 'created logger for ' + name)

    @exc_report
    def log(self, level, msg):
        if isinstance(level, six.string_types):
            level = getattr(logging, level.upper())
        self._logger.log(level, msg)

    def debug(self, msg):
        return self.log(logging.DEBUG, msg)

    def info(self, msg):
        return self.log(logging.INFO, msg)

    def warn(self, msg):
        return self.log(logging.WARN, msg)

    def error(self, msg):
        return self.log(logging.ERROR, msg)

    def critical(self, msg):
        return self.log(logging.CRITICAL, msg)
