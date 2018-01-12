#
# Python support
#
project(kwiver_python)

# Add python modules
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/apm.py
  vital
  apm )

sprokit_add_python_library(config_block
  vital
  config_block.cxx)
target_link_libraries(python-vital-config_block
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_create_python_init(vital
  config_block
)

add_subdirectory( algo )
add_subdirectory( bin ) ##+ do we really need this (maptk application)
add_subdirectory( exceptions )
add_subdirectory( types )
add_subdirectory( tests )
add_subdirectory( util )
