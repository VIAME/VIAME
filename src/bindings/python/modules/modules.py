#!/usr/bin/env python
#ckwg +5
# Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


try:
    from straight.plugin import loaders
except:
    from . import loaders


def log(msg):
    import sys
    sys.stderr.write("%s\n" % msg)
    sys.stderr.flush()


def load_python_module(mod):
    if hasattr(mod, '__vistk_register__'):
        import collections

        if isinstance(mod.__vistk_register__, collections.Callable):
            mod.__vistk_register__()


def load_python_modules():
    import os

    packages = [ 'vistk.processes'
               , 'vistk.schedules'
               ]

    envvar = 'VISTK_PYTHON_MODULES'

    if envvar in os.environ:
        extra_modules = os.environ[envvar]
        packages += extra_modules.split(os.pathsep)

    loader = loaders.ModuleLoader()
    all_modules = []

    for package in packages:
        modules = loader.load(package)

        all_modules += modules

    for module in all_modules:
        try:
            load_python_module(module)
        except BaseException as e:
            log("Failed to load '%s': %s" % (module, str(e)))
