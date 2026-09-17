# ckwg +28
# Copyright 2012-2015 by Kitware, Inc.
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
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from __future__ import print_function, absolute_import
import importlib

from viame.modules import loaders
from viame.util.entrypoint import get_python_plugins_from_entrypoint
from viame import log

logger = log.getLogger(__name__)

MAGIC_REGISTRARS = ["__sprokit_register__", "__vital_algorithm_register__"]


@log.exc_report
def _load_python_module(mod):
    logger.debug('Loading python module: "{}"'.format(mod))
    for registrar in MAGIC_REGISTRARS:
        if hasattr(mod, registrar):
            if callable(getattr(mod, registrar)):
                getattr(mod, registrar)()
                return
            else:
                logger.warn(
                    ('Python module "{}" defined {} but ' "it is not callable").format(
                        mod, registrar
                    )
                )

    logger.debug(('Python module "{}" does not have registrar method').format(mod))


@log.exc_report
def load_python_modules():
    """
    Loads python plugins

    The packages are the ones VIAME ships plus anything named in
    `VIAME_PYTHON_PLUGINS`; see `viame.plugins.discovery`. A package
    that declares what it provides is registered from the declaration; one
    that does not is scanned, which means importing every module in it and
    calling the registrar hook each one defines.
    """
    import os

    from viame.plugins.discovery import (
        LOADED_PACKAGES_ENV_VAR,
        package_declares,
        plugin_packages,
        register_declared_processes,
    )

    logger.info("Loading python modules")

    packages = list(plugin_packages())
    logger.debug(
        "Preparing to load sprokit python plugin modules: "
        "[\n    {}\n]".format(",\n    ".join(list(map(repr, packages))))
    )

    loader = loaders.ModuleLoader()
    all_modules = []
    loaded = []

    # A package that declares what it provides is not scanned. Scanning means
    # importing every module in it that defines a registrar, and the only
    # reason to do that is so the implementation classes exist for the
    # subclass walk in `viame.plugins.discovery`. A declaration gives
    # discovery the same information without the import -- which for
    # `viame.pytorch` is the difference between paying for torch on every
    # command and not.
    for package in packages:
        if package_declares(package):
            logger.debug(
                "Not scanning {}: it declares what it provides".format(package))

            # Declared processes are registered here, with a constructor
            # that imports when a pipeline first wants one. The package
            # itself is still asked to register anything it has not
            # declared; importing it is cheap, which is the discipline a
            # declaration depends on.
            register_declared_processes(package)

            try:
                all_modules.append(importlib.import_module(package))
                loaded.append(package)
            except ImportError as error:
                logger.warn(
                    'Could not import declaring package "{}": {}'.format(
                        package, error))
            continue

        modules = loader.load(package)
        if modules:
            loaded.append(package)
        all_modules += modules

    # So that `registry-dump` can report the packages that contributed, now
    # that no environment variable names them. Setting it here rather than
    # returning it because the reader is C++ in the same process.
    os.environ[LOADED_PACKAGES_ENV_VAR] = os.pathsep.join(loaded)

    all_modules.extend(get_python_plugins_from_entrypoint())

    for module in all_modules:
        try:
            _load_python_module(module)
        except BaseException as ex:
            logger.warn('Failed to load "{}": {}'.format(module, ex))
