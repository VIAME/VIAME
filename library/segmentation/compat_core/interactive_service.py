# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""`python -m viame.core.interactive_service`, as DIVE starts it.

The service is `viame.segmentation.interactive_service` since P2-T05. This
runs that module as `__main__` with the same arguments, so DIVE's command line
keeps working unchanged until DIVE names the new path.
"""

import runpy

if __name__ == "__main__":
    runpy.run_module("viame.segmentation.interactive_service",
                     run_name="__main__", alter_sys=True)
