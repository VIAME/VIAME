# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The python package name VIAME had before phase 11, for one release.

P11-T02 made the whole tree one package: `kwiver.vital.types` became
`viame.types`, `kwiver.vital.algo` became `viame.algo`, and
`kwiver.sprokit.pipeline` became `viame.pipeline`. Pipelines, add-ons, DIVE
and people's own scripts still write the old names, so importing one keeps
working:

    from kwiver.vital.types import ImageContainer   # viame.types

Each old name resolves to the module at its new one, through the same
finder `viame.utilities.module_aliases` installs for the packages phase 2
dissolved. The **same module object** is returned, so a class imported by
either name is one class, and `isinstance` and plugin registration see one
type rather than two that merely look alike.

`aliases.py` is generated from the rename's own table -- see
`design/scripts/build_kwiver_aliases.py` -- so the two cannot drift.

This ships in the release phase 11 lands in and is removed in the next one,
alongside `viame/compat/kwiver.h`, which does the same for the C++
namespaces. Nothing inside VIAME imports it.
"""

from viame.utilities.module_aliases import install as _install

from .aliases import ALIASES

_install( ALIASES )

# `kwiver/__init__.py` defined these two at the top level, and a package
# that registers plugins reads them from here. The values are the new group
# names; `viame.plugins.discovery` reads the old ones as well, for the same
# one release this package lasts.
PYTHON_PLUGIN_ENTRYPOINT = "viame.python_plugin_registration"
CPP_SEARCH_PATHS_ENTRYPOINT = "viame.cpp_search_paths"

__all__ = [ "ALIASES", "PYTHON_PLUGIN_ENTRYPOINT",
            "CPP_SEARCH_PATHS_ENTRYPOINT" ]
