# VIAME Tools Tests

if( VIAME_ENABLE_PYTHON )
  viame_add_pytest_test(
    NAME    "tools:viame_applets"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_viame_applets.py"
    LABELS  tools
    TIMEOUT 900
    SOURCE_SETUP
  )
endif()
