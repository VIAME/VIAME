# VIAME utilities tests

project( test_viame_utilities )

include( viame-test-setup )

if( VIAME_ENABLE_PYTHON )
  # The compatibility packages phase 2 left behind: every old module path
  # must name a module that exists, and import as the same module object.
  viame_add_pytest_test(
    NAME    "unit:utilities:module_aliases"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_module_aliases.py"
    LABELS  UNIT
    TIMEOUT 300
    SOURCE_SETUP
  )
endif()
