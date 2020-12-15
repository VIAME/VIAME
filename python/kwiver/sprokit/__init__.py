# -*- coding: utf-8 -*-
"""
The base SPROKIT package initialization
"""
# flake8: noqa
from __future__ import print_function, unicode_literals, absolute_import

# Configure logging and initialize the root sprokit logger
from kwiver.sprokit import sprokit_logging
sprokit_logging._configure_logging()

logger = sprokit_logging.getLogger(__name__)
# logger.debug('initializing the sprokit python module')
