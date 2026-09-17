# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Environment variables that were renamed in phase 11.

The C++ side reads these pairs through `viame::get_env_renamed`; this is the
same rule for python. The name VIAME uses now wins; the name it used before
is still read, because an environment written for an older VIAME sets it and
would otherwise change behaviour with nothing said. Using the old name is
reported once per process at warning level -- failing on it would break every
such environment, and ignoring it quietly turns "my setting does nothing"
into a puzzle with no way in.
"""

import logging
import os

__all__ = ["get_renamed"]

_LOG = logging.getLogger(__name__)

# The old names already reported, so that a variable read on every call --
# the plugin search path is read once per discovery pass -- says it once.
_WARNED = set()


def get_renamed(name, old_name, default=None):
    """The value of `name`, else of `old_name`, else `default`.

    Reading `old_name` is warned about the first time only.
    """
    value = os.environ.get(name)
    if value is not None:
        return value

    old_value = os.environ.get(old_name)
    if old_value is None:
        return default

    if old_name not in _WARNED:
        _WARNED.add(old_name)
        _LOG.warning(
            "%s is set. It is the name VIAME used before phase 11, is still "
            "read, and will stop being read in a later release; set %s "
            "instead.", old_name, name)

    return old_value
