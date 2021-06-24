
# All files which are apart of kwiver plugins and their dependencies
set( _rel_modnames
  algos
  demo
  expt
  imutils
  processes
  util_algo
  __init__
  __main__
  )

foreach( rel_modname IN ITEMS ${_rel_modnames} )
  string( REPLACE "." "/" rel_modpath_noext ${rel_modname} )
  kwiver_add_python_module( "${CMAKE_CURRENT_SOURCE_DIR}/${rel_modpath_noext}.py"
    "viame/processes/camtrawl" "${rel_modname}" )
endforeach()
