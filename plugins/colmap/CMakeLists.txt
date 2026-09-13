###
# COLMAP structure-from-motion / dense reconstruction + survey-registration
# plugin (Python-only).
#
# Provides viame.colmap.reconstruction (used by the 3d tool for the
# non-planar SfM / dense modes; requires pycolmap + open3d) and the
# colmap_registration sprokit process (multi-camera survey registration; no
# pycolmap dependency). Its __init__ declares the process; discovery registers
# it with a constructor that imports this package's modules on first use.
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
    viame_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/${modname}.py
      colmap
      ${modname} )
  endforeach()
endif()
