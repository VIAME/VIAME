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

kwiver_install_headers(
  kwiver_type_traits.h
  trait_utils.h
  )
