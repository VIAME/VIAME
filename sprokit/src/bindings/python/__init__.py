# -*- coding: utf-8 -*-
"""
The base SPROKIT package initialization
"""
from __future__ import print_function, unicode_literals, absolute_import
import sys
from sprokit import sprokit_logging

sprokit_logging._configure_logging()

logger = sprokit_logging.SprokitLogger(__name__)
logger.log('debug', 'initializing sprokit module')


# Handle issues when called from pybind11 modules
@sprokit_logging.exc_report
def _pybind11_argv_workaround():
    """
    If this module is imported by pybind11, argv will not be populated.  This
    can cause issues with other modules, so set it as an attribute here.
    """
    if not hasattr(sys, 'argv'):
        sys.argv = []


@sprokit_logging.exc_report
def _pybind11_excepthook_workaround():
    """
    Implement a custom excepthook to (try and) ensure python errors are
    reported.

    Currently, I'm unsure if this even does anything. Pybind11 might just not
    care about excepthooks.
    """
    _sys_excepthook = getattr(sys, 'excepthook', None)

    def sprokit_excepthook(*exc_info):
        print('! SPROKIT EXCEPTHOOK WAS TRIGGERED !')
        try:
            sprokit_logging.print_exc(exc_info)
        except Exception:
            if _sys_excepthook is not None:
                _sys_excepthook(*exc_info)

    sys.excepthook = sprokit_excepthook

_pybind11_argv_workaround()
_pybind11_excepthook_workaround()
