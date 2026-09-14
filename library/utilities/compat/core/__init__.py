# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""`viame.core`, kept so that its old module paths still import.

the `core` plugin was dissolved into the libraries in `library/` in phase 2.
Every module it installed is aliased to where it lives now -- the table is
`aliases.py`, generated from `design/lite-file-map.tsv` by
`design/scripts/build_compat_aliases.py` -- so `import viame.core.x`,
`from viame.core.x import Y` and `python -m viame.core.x` all reach
the moved module, as the same module object. Nothing is imported until an old
name is.

Registration is not here: the libraries' own packages declare everything, and
this package declaring nothing keeps discovery from scanning it.
"""

from viame.utilities.module_aliases import install as _install

from .aliases import ALIASES as _ALIASES

_install(_ALIASES)

__vital_algorithm_declarations__ = []
__sprokit_process_declarations__ = []
