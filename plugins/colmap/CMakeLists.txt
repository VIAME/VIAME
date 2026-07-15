###
# COLMAP structure-from-motion / dense reconstruction + survey-registration
# plugin (Python-only).
#
# Provides viame.colmap.reconstruction (used by the reconstruct_3d tool for the
# non-planar SfM / dense modes; requires pycolmap + open3d) and the
# colmap_registration sprokit process (multi-camera survey registration; no
# pycolmap dependency). This package is registered as a sprokit python process
# module via SPROKIT_PYTHON_MODULES (see the setup template), and its custom
# __init__ registers the process.
##

if( VIAME_ENABLE_PYTHON )
  # A custom __init__ (with __sprokit_register__) instead of the generated
  # empty one, so this package can host sprokit processes.
  set( colmap_modules
    __init__
    reconstruction
    prior_coverage_sfm
    colmap_registration )

  foreach( modname IN ITEMS ${colmap_modules} )
    kwiver_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/${modname}.py
      colmap
      ${modname} )
  endforeach()
endif()
