
# flake8: noqa
from __future__ import print_function, unicode_literals, absolute_import

# Configure logging and initialize the root sprokit logger
from kwiver.vital import vital_logging
vital_logging._configure_logging()

logger = vital_logging.getLogger(__name__)
