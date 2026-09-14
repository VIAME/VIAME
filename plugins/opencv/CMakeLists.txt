
###
# Python
##

# No library left: P2-T05 took the filters, detectors and segmenter, P2-T06
# the stereo pairing and measurement, and P2-T07 the windowed trainer, merged
# with core's as `library/training`'s `windowed` (`ocv_windowed` an alias).
# What is left is the stereo demo scripts lite-library-layout.md deletes with
# the directory in P2-T08.

if( VIAME_ENABLE_PYTHON )
  set( _rel_modnames
    __init__
    stereo_demo
    stereo_expt
  )

  foreach( rel_modname IN ITEMS ${_rel_modnames} )
    string( REPLACE "." "/" rel_modpath_noext ${rel_modname} )
    viame_add_python_module(
      "${CMAKE_CURRENT_SOURCE_DIR}/${rel_modpath_noext}.py"
      "opencv"
      "${rel_modname}" )
  endforeach()
endif()

