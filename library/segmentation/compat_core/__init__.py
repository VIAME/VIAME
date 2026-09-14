# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The `viame.core` module paths something outside this tree still runs.

`viame.core` was `plugins/core`'s package, and P2-T04 to P2-T07 moved all of
it into the libraries in `library/`. DIVE's desktop client starts
`python -m viame.core.interactive_service`, both at the commit `packages/dive`
pins and on DIVE's main branch, so that one path is kept as a forwarder to
where the module lives now. Nothing here registers anything.
"""

__vital_algorithm_declarations__ = []
__sprokit_process_declarations__ = []
