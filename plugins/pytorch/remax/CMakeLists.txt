###
# Loadable algorithms
viame_create_python_init( pytorch/remax )

viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/ReMax.py
      pytorch/remax
      ReMax)

add_subdirectory( util )
add_subdirectory( model )
