# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""viame.colmap - COLMAP / survey-registration plugin.

Hosts the colmap_registration sprokit process (available whenever this
package is on SPROKIT_PYTHON_MODULES). The heavier SfM / dense-reconstruction
modules (reconstruction, prior_coverage_sfm) require pycolmap and are imported
lazily by their callers, so process registration never pulls them in.
"""


def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:viame.colmap.colmap_processes'
    if process_factory.is_process_module_loaded(module_name):
        return

    # The registration node depends only on viame.opencv (no pycolmap), so it
    # registers even when COLMAP's optional dependencies are absent.
    try:
        from viame.colmap import colmap_registration
        colmap_registration.__sprokit_register__()
    except ImportError:
        pass

    process_factory.mark_process_module_as_loaded(module_name)
