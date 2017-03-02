#
# Kwiver processes
#

include_directories( ${sprokit_source_dir}/src
                     ${sprokit_binary_dir}/src   # for generated files
                     ${CMAKE_CURRENT_SOURCE_DIR} # for trait support files
                     )

add_subdirectory( adapters )
add_subdirectory( bindings )
add_subdirectory( core )
add_subdirectory( examples )

if (KWIVER_ENABLE_VXL)
  add_subdirectory( vxl )
endif()

if (KWIVER_ENABLE_OPENCV)
  add_subdirectory( ocv )
endif()

if (KWIVER_ENABLE_PYTHON)
  add_subdirectory( python )
endif()

if (KWIVER_ENABLE_CAFFE)
#  add_subdirectory(caffe)
endif()

if (KWIVER_ENABLE_MATLAB)
  add_subdirectory( matlab )
endif()

kwiver_install_headers(
  kwiver_type_traits.h
  trait_utils.h
  SUBDIR sprokit/processes
  )

if (KWIVER_ENABLE_TOOLS)
###
#     plugins for plugin explorer
kwiver_add_plugin( process_explorer_plugin
  SUBDIR   modules/plugin_explorer
  SOURCES  process_explorer_plugin.cxx
  PRIVATE  vital
           vital_vpm
           vital_config
           vital_algo
           kwiversys
           sprokit_pipeline
           explorer_plugin )
endif()
