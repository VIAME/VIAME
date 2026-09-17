# core_types tests

project( test_viame_core_types )

include( viame-test-setup )

viame_discover_gtests( viame math
  LIBRARIES kwiver::vital
            GTest::gtest )

# The recording the math tests read. The source tree rather than a copy, so
# that regenerating it with tests/golden/math/record_from_eigen.cxx takes
# effect without a reconfigure.
target_compile_definitions( test-viame-math PRIVATE
  "VIAME_GOLDEN_MATH_DIR=\"${VIAME_SOURCE_DIR}/tests/golden/math\"" )

if( VIAME_ENABLE_PYTHON )
  # The python API `viame.types` presents, which P8-T01 rewrites: what
  # it exposes, held against `tests/golden/python_types.json`, and what it
  # does. See the test file's own note for why it needs both halves.
  add_test(
    NAME viame_library:core_types:python_types
    COMMAND bash -c
      "source ${VIAME_BUILD_INSTALL_PREFIX}/setup_viame.sh && \
       python -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_python_types.py \
       -v --tb=short"
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
  set_tests_properties( viame_library:core_types:python_types
    PROPERTIES
      LABELS "UNIT;CORE"
      TIMEOUT 300
  )
endif()
