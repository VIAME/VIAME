#!/usr/bin/env python
#ckwg +5
# Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def log(msg):
    import sys
    sys.stderr.write("%s\n" % msg)


def load_python_module(mname):
    mod = __import__(mname)

    import sys
    mod = sys.modules[mname]

    if hasattr(mod, '__vistk_register__'):
        if callable(mod.__vistk_register__):
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

    modules = []

    while packages:
        import pkgutil

        pname = packages.pop()
        try:
            pkg = __import__(pname)
        except BaseException as e:
            log("Failed to import '%s': '%s'" % (pname, str(e)))
            continue

        modules.append(pname)

        import sys
        pkg = sys.modules[pname]

        prefix = pkg.__name__ + "."
        for _, modname, ispkg in pkgutil.iter_modules(pkg.__path__, prefix):
            if ispkg:
                packages.append(modname)
            else:
                modules.append(modname)

    for module in modules:
        #try:
            load_python_module(module)
        #except BaseException as e:
            #log("Failed to load '%s': %s" (module, str(e)))
