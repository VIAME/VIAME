###
# Loadable algorithms

if( VIAME_ENABLE_PYTHON )
  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    viame/processes/template
    __init__)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/template_detector.py
    viame/processes/template
    template_detector)

endif()
