if( VIAME_ENABLE_PYTHON )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    viame/processes/smqtk
    __init__ )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/smqtk_add_descriptors_uuids.py
    viame/processes/smqtk
    smqtk_add_descriptors_uuids )

endif()
