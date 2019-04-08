# -*- coding: utf-8 -*-
"""
A thin wrapper around python logging.
"""
from __future__ import print_function, unicode_literals, absolute_import
import sys
import os
import logging


def _configure_logging():
    """
    Configures python logging to using KWIVER / SPROKIT environment variables

    SeeAlso:
        kwiver/vital/logger: logic for the vital logger
    """
    # Use the C++ logging level by default, but allow python to be different
    cxx_level = os.environ.get('KWIVER_DEFAULT_LOG_LEVEL', 'DEBUG')
    py_level = os.environ.get('KWIVER_PYTHON_DEFAULT_LOG_LEVEL', cxx_level)

    # Option to colorize the python logs (must pip install coloredlogs)
    truthy_values = {'true', 'on', 'yes', '1'}
    use_color_env = os.environ.get('KWIVER_PYTHON_COLOREDLOGS', 'false')

    # Default options
    use_color = use_color_env.strip().lower() in truthy_values
    level = getattr(logging, py_level.upper())
    # Match KWIVER's log prefix: date time level file(lineno)
    logfmt = '%(asctime)s.%(msecs)03d %(levelname)s %(name)s(%(lineno)d): %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    # Maybe use file based configs in the future?

    if use_color:
        import coloredlogs
        # The colorscheme can be controlled by several environment variables
        # https://coloredlogs.readthedocs.io/en/latest/#environment-variables
        coloredlogs.install(level=level, fmt=logfmt, datefmt=datefmt)
    else:
        logging.basicConfig(format=logfmt, level=level, datefmt=datefmt)


def print_exc(exc_info=None):
    """
    Prints a highly visible exception.

    Args:
        exc_info (tuple): result of `sys.exec_info()`

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

    Modules called from C++ (e.g. pybind11) are not guaranteed to print
    exceptions if they occur (unless the C++ program does the work). This
    decorator can be used as a workaround to ensure that there is some
    indication of when Python code raises an exception.

    Args:
        func (callable): function to dectorate
    """
    def _exc_report_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            print_exc()
            raise
    return _exc_report_wrapper


@exc_report
def getLogger(name):
    """
    Lighweight wrapper around python's logging.getLogger function.

    Args:
        name (str): Logger name, which should be the module __name__ attribute

    Example:
        >>> from sprokit import sprokit_logging
        >>> logger = sprokit_logging.getLogger(__name__)
        >>> logger.info('Hello World')

    This really should get a vital logger rather than a python logger.
    The vital loggers use the logger back-end and can be configured by a file.
    """
    _logger = logging.getLogger(name)
    # _logger.debug('created logger for ' + name)
    return _logger
