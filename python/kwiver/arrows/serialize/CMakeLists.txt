if ( KWIVER_ENABLE_SERIALIZE_JSON )
  add_subdirectory( json )
endif()

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                                arrows/serialize
                                __init__)
