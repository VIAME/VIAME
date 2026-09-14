# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Helpers the rest of VIAME imports.

`vital_registration` is the base every python algorithm derives from,
`compat` and `types` smooth over what the interpreter and kwiver provide,
and `utils`, `utilities_ply` and `model_wrap` are what their names say.

Nothing here registers an algorithm or a process, so there are no
`__vital_algorithm_declarations__` and no `__sprokit_process_declarations__`
and this package is deliberately absent from `BUILTIN_PLUGIN_PACKAGES`.
"""
