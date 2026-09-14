# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Setting a detection's mask.

`interactive_segmentation` is the tool DIVE drives, `interactive_service`
the host it talks to, and `segmentation_utils` what both build on. None of
them registers an algorithm or a process -- they are run directly, as
`python -m viame.segmentation.interactive_segmentation` -- so this package
has no declarations and is deliberately absent from
`BUILTIN_PLUGIN_PACKAGES`.
"""
