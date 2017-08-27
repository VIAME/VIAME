if( VIAME_ENABLE_PYTHON AND VIAME_ENABLE_SMQTK )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    viame/processes/smqtk
    __init__ )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/smqtk_ingest_descriptors.py
    viame/processes/smqtk
    smqtk_ingest_descriptors )
endif()
