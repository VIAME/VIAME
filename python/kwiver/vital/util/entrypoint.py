# ckwg +29
# Copyright 2020 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF


from kwiver.vital import vital_logging
from kwiver.vital.plugins.discovery import get_ns_entrypoints
from kwiver import PYTHON_PLUGIN_ENTRYPOINT


logger = vital_logging.getLogger(__name__)


def get_python_plugins_from_entrypoint():
    """
    Get a list of python plugins that were registered through
    kwiver.python_plugin_registration
    :return: A list of zero or more python modules containing registration
             functions

    This reads the same entry points `pkg_resources.iter_entry_points` did,
    through `importlib.metadata` instead. `pkg_resources` costs 110 ms to
    import -- a fifth of the startup budget P8-T10 sets, on every command,
    to look for a hook that a stock install has no users of. The stdlib
    reader is free, and `kwiver.vital.plugins.discovery` was already using
    it for the other entry point group.
    """
    py_modules = []
    for entry_point in get_ns_entrypoints(PYTHON_PLUGIN_ENTRYPOINT):
        try:
            py_modules.append(entry_point.load())
        except ImportError:
            logger.warn("Failed to load entry point: {0}".format(entry_point))
    return py_modules
