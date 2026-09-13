###
# Loadable algorithms

if( VIAME_ENABLE_PYTHON )
  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/@template@_detector.py
    @template_dir@
    @template@_detector)
endif()
