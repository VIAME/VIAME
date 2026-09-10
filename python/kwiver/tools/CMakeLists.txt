kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  tools
  __init__  )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/dump_klv.py
  tools
  dump_klv)
configure_file(
  dump_klv.py
  ${KWIVER_BINARY_DIR}/bin/dump_klv.py
  COPYONLY
)


if( KWIVER_ENABLE_PYTHON_TESTS )
  add_subdirectory(tests)
endif()
