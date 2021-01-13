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


from pkg_resources import iter_entry_points, DistributionNotFound
from kwiver.vital import vital_logging
from kwiver import PYTHON_PLUGIN_ENTRYPOINT, CPP_SEARCH_PATHS_ENTRYPOINT
import kwiver

import os



logger = vital_logging.getLogger(__name__)

def get_python_plugins_from_entrypoint():
    """
    Get a list of python plugins that were registered through
    kwiver.python_plugin_registration
    :return: A list of zero or more python modules containing registration
             functions
    """
    py_modules = []
    try:
        for entry_point in iter_entry_points(PYTHON_PLUGIN_ENTRYPOINT):
            try:
                py_modules.append(entry_point.load())
            except ImportError:
                logger.warn("Failed to load entry point: {0}".format(entry_point))
    except DistributionNotFound:
        pass
    return py_modules


def get_cpp_paths_from_entrypoint():
    """
    Get a list of paths that were advertised through kwiver.cpp_search_paths
    :return: A list of paths for c++ plugins
    """
    additional_search_paths = []
    try:
        for entry_point in iter_entry_points(CPP_SEARCH_PATHS_ENTRYPOINT):
            try:
                search_path = entry_point.load()()
            except ImportError:
                logger.warn("Failed to load entry point: {0}".format(entry_point))
                continue

            if os.path.exists(search_path):
                additional_search_paths.append(search_path)
            else:
                logger.warn('Invalid search path {0} specified by {1}'.format(search_path,
                            entry_point))
    except DistributionNotFound:
        pass

    return additional_search_paths

def add_entrypoint_paths_to_env():
    additional_search_paths = get_cpp_paths_from_entrypoint()
    current_ld_path = os.getenv("LD_LIBRARY_PATH", "")
    new_ld_path = current_ld_path
    for additional_search_path in additional_search_paths:
        new_ld_path += ":{0}".format(additional_search_path)
    os.environ['LD_LIBRARY_PATH'] = new_ld_path

def get_library_path():
    return os.path.join(os.path.dirname(os.path.abspath(kwiver.__file__)),
           'lib')

def get_vital_logger_factory():
    return os.path.join(os.path.dirname(os.path.abspath(kwiver.__file__)),
           'lib', 'kwiver', 'modules', 'vital_log4cplus_logger')

def sprokit_process_path():
    return os.path.join(os.path.dirname(os.path.abspath(kwiver.__file__)),
           'lib', 'kwiver', 'processes')

def applets_path():
    return os.path.join(os.path.dirname(os.path.abspath(kwiver.__file__)),
           'lib', 'kwiver', 'modules', 'applets')

def plugin_explorer_path():
    return os.path.join(os.path.dirname(os.path.abspath(kwiver.__file__)),
           'lib', 'kwiver', 'modules', 'plugin_explorer')
