#ckwg +4
# Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


try:
    from . import loaders
except:
    from straight.plugin import loaders


def _log(msg):
    import sys
    sys.stderr.write("%s\n" % msg)
    sys.stderr.flush()


def _load_python_module(mod):
    if hasattr(mod, '__sprokit_register__'):
        import collections

        if isinstance(mod.__sprokit_register__, collections.Callable):
            mod.__sprokit_register__()


def load_python_modules():
    import os

    packages = [ 'sprokit.processes'
               , 'sprokit.schedulers'
               ]

    envvar = 'SPROKIT_PYTHON_MODULES'

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
            _load_python_module(module)
        except BaseException:
            import sys

            e = sys.exc_info()[1]

            _log("Failed to load '%s': %s" % (module, str(e)))
