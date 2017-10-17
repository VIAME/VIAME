# -*- coding: utf-8 -*-
"""
The base SPROKIT package initialization
"""
# flake8: noqa
from __future__ import print_function, unicode_literals, absolute_import
import sys


# Handle issues when called from pybind11 modules
def _pybind11_argv_workaround():
    """
    If this module is imported by pybind11, argv will not be populated.  This
    can cause issues with other modules, so set it as an attribute here.
    """
    if not hasattr(sys, 'argv'):
        sys.argv = []
_pybind11_argv_workaround()


# Configure logging and initialize the root sprokit logger
from sprokit import sprokit_logging
sprokit_logging._configure_logging()

logger = sprokit_logging.getLogger(__name__)
logger.debug('initializing the sprokit python module')
