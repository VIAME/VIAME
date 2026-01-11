###
# Loadable algorithms
kwiver_create_python_init( pytorch/remax )

kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/ReMax.py
      pytorch/remax
      ReMax)

add_subdirectory( util )
add_subdirectory( model )
