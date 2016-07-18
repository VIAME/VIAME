"""
ckwg +31
Copyright 2015-2016 by Kitware, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Kitware, Inc. nor the names of any contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==============================================================================

Interface to VITAL algorithm_plugin_manager class.

"""
# -*- coding: utf-8 -*-
import ctypes
from vital.util import VitalObject


__author__ = 'paul.tunison@kitware.com'


def register_plugins(name=None):
    """
    (Re)Load plugin modules found in currently set search paths.

    If a plugin name was provided, we attempt to load only that plugin
    module. If the named plugin fails to load, nothing occurs (warning
    messages will be emitted, though).

    :param name: Optional name of a specific plugin.
    :type name: str

    """
    if name is None:
        apm_reg_plugins = VitalObject.VITAL_LIB.vital_apm_register_plugins
        apm_reg_plugins()
    else:
        apm_reg_plugins = VitalObject.VITAL_LIB.vital_apm_register_single_plugin
        apm_reg_plugins.argtypes = [ctypes.c_char_p]
        apm_reg_plugins(name)


def register_plugins_once():
    """
    Load all plugins on first call.

    This static method loads all plugins on the first call and does
    nothing on all subsequent calls. This is designed to load plugins
    in a concurrent application where the first thread to start is
    non-deterministic. All threads would call this method on starting
    and the first one that completes has loaded all plugins and the
    other callers will return.

    If you must reload plugins after this method has been called, use
    the vital_apm_register_plugins() method.

    :return: True if plugins were loaded, False if plugins were already
        loaded.
    :rtype: bool

    """
    apm_load_once = VitalObject.VITAL_LIB['vital_apm_load_plugins_once']
    apm_load_once.restype = bool
    return apm_load_once()


def add_search_path(dirpath):
    """
    Add an additional directory to search for plugins in

    :param dirpath: Path to a directory to additionally search for plugins
    :type dirpath: str

    """
    apm_add_sp = VitalObject.VITAL_LIB.vital_apm_add_search_path
    apm_add_sp.argtypes = [ctypes.c_char_p]
    apm_add_sp(dirpath)


def registered_module_names():
    """ Get a list of registered module name strings

    A module's name is defined as the filename minus the standard platform
    module library suffix. For example, on Windows, if a module library was
    named ``vital_foo.dll``, the module's name would be "vital_foo".
    Similarly on a unix system, ``vital_bar.so`` would have the name
    "vital_bar".

    :return: List of registered module names
    :rtype: list of str

    """
    apm_reg_names = VitalObject.VITAL_LIB.vital_apm_registered_module_names
    sl_free = VitalObject.VITAL_LIB.vital_common_free_string_list

    apm_reg_names.argtypes = [ctypes.POINTER(ctypes.c_uint),
                              ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p))]
    sl_free.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_char_p)]

    length = ctypes.c_uint(0)
    keys = ctypes.POINTER(ctypes.c_char_p)()
    apm_reg_names(ctypes.byref(length), ctypes.byref(keys))

    # Constructing return array
    r = []
    for i in xrange(length.value):
        r.append(keys[i])

    # Free allocated key listing
    sl_free(length, keys)

    return r
