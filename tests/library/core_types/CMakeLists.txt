# core_types tests

project( test_viame_core_types )

include( kwiver-test-setup )

kwiver_discover_gtests( viame math
  LIBRARIES kwiver::vital
            GTest::gtest )

# The recording the math tests read. The source tree rather than a copy, so
# that regenerating it with tests/golden/math/record_from_eigen.cxx takes
# effect without a reconfigure.
target_compile_definitions( test-viame-math PRIVATE
  "VIAME_GOLDEN_MATH_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/../../golden/math\"" )
