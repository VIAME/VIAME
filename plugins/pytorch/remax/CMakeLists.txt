###
# Loadable algorithms
kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/ReMax.py
      arrows/pytorch/remax
      ReMax)

add_subdirectory( util )
add_subdirectory( model )
