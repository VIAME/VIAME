#
# Kwiver processes
#

# This policy is new in CMake 3.12. The NEW behavior uses the <PackageName>_ROOT variable in
# find_package(<PackageName>) calls.
# See: https://cmake.org/cmake/help/git-stage/policy/CMP0074.html
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

include_directories( ${sprokit_source_dir}/src
                     ${sprokit_binary_dir}/src   # for generated files
                     ${CMAKE_CURRENT_SOURCE_DIR} # for trait support files
                     )

add_subdirectory( adapters )
add_subdirectory( bindings )
add_subdirectory( core )
add_subdirectory( examples )
add_subdirectory( flow )

#
# This may need some enable flags as we start to support other
# transports that require support libraries such as zeromq.
#
add_subdirectory( transport )

if (KWIVER_ENABLE_VXL)
  add_subdirectory( vxl )
endif()

if (KWIVER_ENABLE_OPENCV)
  add_subdirectory( ocv )
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
  SUBDIR   ${kwiver_plugin_plugin_explorer_subdir}
  SOURCES  process_explorer_plugin.cxx
  PRIVATE  vital
           vital_vpm
           vital_config
           vital_algo
           kwiversys
           sprokit_pipeline
           explorer_plugin )
endif()

if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()
