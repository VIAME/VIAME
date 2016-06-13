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

Utility method for finding and returning the VITAL library.
e.g. libvital_c.so

"""
# -*- coding: utf-8 -*-
__author__ = 'paul.tunison@kitware.com'

import ctypes
import os
import re
import sys


__LIBRARY_NAME__ = "vital_c"
__LIBRARY_NAME_RE_BASE__ = "(?:lib)?%s.(?:so|dylib|dll).*"
__LIBRARY_NAME_RE__ = re.compile(__LIBRARY_NAME_RE_BASE__ % __LIBRARY_NAME__)
__LIBRARY_PATH_CACHE__ = None
__LIBRARY_CACHE__ = None


def _system_library_dirs():
    """
    :return: Platform dependent relative paths to directories containing library
        files.
    :rtype: str

    """
    if sys.platform == "win32":
        return ["bin"]
    else:
        return ["lib64", "lib"]


def _search_up_directory(d, library_re):
    """
    Look for the vital_c library in the given directory and each directory above
    it until we hit root. We return the first matching library path, or None
    if no matching library was found.

    :param d: Starting directory path
    :param library_re: regular expression for library
    :type d: str

    :return: Discovered path to the vital_c library, or None if one wasn't found
    :rtype: str or None

    """
    # Stopping search when we see the root twice, based on ``os.path.dirname``
    # returning the root directory when the root directory is passed to it.
    prev_dir = None
    d = os.path.abspath(d)
    while d != prev_dir:
        for l in _system_library_dirs():
            l_dir = os.path.join(d, l)
            if os.path.isdir(l_dir):
                for f in os.listdir(l_dir):
                    if library_re.match(f):
                        return os.path.join(l_dir, f)
        prev_dir = d
        d = os.path.dirname(d)
    return None


def _system_path_separator():
    """
    System dependent character for element separation in PATH variable
    :rtype: str
    """
    if sys.platform == 'win32':
        return ';'
    else:
        return ':'


def find_vital_library_path(use_cache=True):
    """
    Discover the path to a VITAL C interface library based on the directory
    structure this file is in, and then to system directories in the
    LD_LIBRARY_PATH.

    :param use_cache: Store and use the cached path, preventing redundant
        searching (default = True).
    :type use_cache: bool

    :return: The string path to the VITAL C interface library
    :rtype: str

    """
    global __LIBRARY_PATH_CACHE__
    if use_cache and __LIBRARY_PATH_CACHE__:
        return __LIBRARY_PATH_CACHE__

    # Otherwise, find the Vital C library
    search_dirs = [os.path.dirname(os.path.abspath(__file__))]
    # NOTE this is not cover all possible systems
    search_dirs.extend(os.environ['LD_LIBRARY_PATH'].split(_system_path_separator()))

    for d in search_dirs:
        r = _search_up_directory(d, __LIBRARY_NAME_RE__)
        if r is not None:
            if use_cache:
                __LIBRARY_PATH_CACHE__ = r
            return r

    # No library found in any paths given at this point
    raise RuntimeError("Failed to find a valid '%s' library!"
                       % __LIBRARY_NAME__)


def find_vital_library(use_cache=True):
    """
    Discover and return the ctypes-loaded VITAL C interface library.

    :param use_cache: Use the cached library instance or cache the discovered
        library. Otherwise, search for the library again, not storing it in the
        cache. Default is True.
    :type use_cache: bool

    :return: The cached Vital C library ctypes instance.
    :rtype: ctypes.CDLL

    """
    if use_cache:
        global __LIBRARY_CACHE__
        if not __LIBRARY_CACHE__:
            __LIBRARY_CACHE__ = ctypes.CDLL(find_vital_library_path(use_cache))
        return __LIBRARY_CACHE__
    else:
        return ctypes.CDLL(find_vital_library_path(use_cache))
