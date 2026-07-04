###
# COLMAP structure-from-motion / dense reconstruction plugin (Python-only).
#
# Provides viame.colmap.reconstruction, used by the reconstruct_3d tool
# for the non-planar SfM / dense (MVS) modes. Requires pycolmap + open3d (added
# to the Python dependencies when VIAME_ENABLE_COLMAP is ON).
##

if( VIAME_ENABLE_PYTHON )
  kwiver_create_python_init( colmap )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction.py
    colmap
    reconstruction )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/prior_coverage_sfm.py
    colmap
    prior_coverage_sfm )
endif()
