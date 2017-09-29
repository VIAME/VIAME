#ckwg +28
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


from __future__ import print_function
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

    else:
        print(('[WARN] Python module {} does not have '
               '__sprokit_register__ method').format(mod))


def load_python_modules():
    """
    Loads Sprokit python plugins

    Searches for modules specified in the `SPROKIT_PYTHON_MODULES` environment
    variable that are importable from `PYTHONPATH`. Then these modules are
    imported and their `__sprokit_register__` function is called to register
    them with the C++ backend.
    """
    import os

    packages = ['sprokit.processes',
                'sprokit.schedulers']

    envvar = 'SPROKIT_PYTHON_MODULES'

    extra_modules = os.environ.get(envvar, '').split(os.pathsep)
    # ensure the empty string is not considered as a module
    packages.extend([p for p in extra_modules if p])

    loader = loaders.ModuleLoader()
    all_modules = []

    for package in packages:
        modules = loader.load(package)
        all_modules += modules

    for module in all_modules:
        print('[DEBUG] Loading python module: {}'.format(module))

        try:
            _load_python_module(module)
        except BaseException:
            import sys
            e = sys.exc_info()[1]
            _log("Failed to load '%s': %s" % (module, str(e)))
