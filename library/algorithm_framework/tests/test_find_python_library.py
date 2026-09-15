# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""`kwiver.vital.util.find_python_library`, which the python module loader
uses to preload libpython when `PYTHON_LIBRARY` is not set.

`setup_viame.sh` stopped setting it in P10-T01, and on Debian and Ubuntu the
finder then returned nothing: their python reports a `LIBDIR` that already
has the multiarch directory in it, and the finder appended that directory
again. Every `viame` run logged an error for it. These fake both layouts that
matter -- a distribution's and a CPython built from source into the install --
over real files, and load the module from the source tree, so nothing about
the machine running the test decides the result.
"""

import importlib.util
import os
import sys
import sysconfig

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SOURCE = os.environ.get(
    "VIAME_FIND_PYTHON_LIBRARY_SOURCE",
    os.path.join(ROOT, "python", "kwiver", "vital", "util", "find_python_library.py"))

# The finder builds its fallback candidate names from the running
# interpreter's version, so a faked library has to carry that version.
VER = "%d.%d" % sys.version_info[:2]
SONAME = "libpython%s.so.1.0" % VER
DEV_LINK = "libpython%s.so" % VER
STATIC = "libpython%s.a" % VER


def load_finder():
    spec = importlib.util.spec_from_file_location("find_python_library_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_config(monkeypatch, finder, values):
    def get_config_var(name):
        return values.get(name)
    monkeypatch.setattr(sysconfig, "get_config_var", get_config_var)
    monkeypatch.setattr(finder.du_sysconfig, "get_config_var", get_config_var)


def touch(path):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    open(str(path), "w").close()
    return str(path)


def test_debian_multiarch_libdir(tmp_path, monkeypatch):
    # Ubuntu's python: LIBDIR already ends in the multiarch directory,
    # multiarchsubdir names it again, and LIBRARY is the static archive.
    libdir = tmp_path / "usr" / "lib" / "x86_64-linux-gnu"
    shared = touch(libdir / SONAME)
    touch(libdir / DEV_LINK)
    finder = load_finder()
    fake_config(monkeypatch, finder, {
        "LIBDIR": str(libdir),
        "LIBPL": str(tmp_path / "usr" / "lib" / ("python" + VER) / "config"),
        "LDLIBRARY": DEV_LINK,
        "INSTSONAME": SONAME,
        "LIBRARY": STATIC,
        "MULTIARCH": "x86_64-linux-gnu",
        "multiarchsubdir": "/x86_64-linux-gnu",
        "Py_ENABLE_SHARED": 1,
    })
    assert finder.find_python_library() == shared


def test_cpython_built_into_the_install(tmp_path, monkeypatch):
    # VIAME_BUILD_PYTHON_FROM_SOURCE: no multiarch, the shared library in the
    # install's own lib/. Either file is the right library; this layout was
    # never the broken one.
    libdir = tmp_path / "install" / "lib"
    shared = touch(libdir / SONAME)
    dev_link = touch(libdir / DEV_LINK)
    finder = load_finder()
    fake_config(monkeypatch, finder, {
        "LIBDIR": str(libdir),
        "LDLIBRARY": DEV_LINK,
        "INSTSONAME": SONAME,
        "LIBRARY": STATIC,
        "MULTIARCH": "",
        "Py_ENABLE_SHARED": 1,
    })
    assert finder.find_python_library() in (shared, dev_link)


def test_static_python_finds_nothing_rather_than_something_wrong(tmp_path, monkeypatch):
    # A python with no shared library at all: the answer is "none", which the
    # loader reports, not a path that does not exist.
    libdir = tmp_path / "lib"
    libdir.mkdir()
    finder = load_finder()
    fake_config(monkeypatch, finder, {
        "LIBDIR": str(libdir),
        "LDLIBRARY": STATIC,
        "LIBRARY": STATIC,
        "MULTIARCH": "",
        "Py_ENABLE_SHARED": 0,
    })
    assert finder.find_python_library() == ""
