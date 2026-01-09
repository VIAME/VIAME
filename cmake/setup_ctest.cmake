# Setup CTest to include viame-build tests
#
# Required variables:
#   VIAME_BUILD_DIR - top-level build directory
#   VIAME_TEST_DIR  - viame-build plugins directory containing tests

if(NOT DEFINED VIAME_BUILD_DIR)
  message(FATAL_ERROR "VIAME_BUILD_DIR is not defined")
endif()

if(NOT DEFINED VIAME_TEST_DIR)
  message(FATAL_ERROR "VIAME_TEST_DIR is not defined")
endif()

# Path to the main CTestTestfile.cmake
set(CTEST_FILE "${VIAME_BUILD_DIR}/CTestTestfile.cmake")

# Check if viame-build tests exist
if(EXISTS "${VIAME_TEST_DIR}/CTestTestfile.cmake")
  # Check if subdirs entry already exists to avoid duplicates
  if(EXISTS "${CTEST_FILE}")
    file(READ "${CTEST_FILE}" CTEST_CONTENTS)
    string(FIND "${CTEST_CONTENTS}" "${VIAME_TEST_DIR}" FOUND_POS)
    if(FOUND_POS EQUAL -1)
      # Append subdirs directive to include viame-build tests
      file(APPEND "${CTEST_FILE}"
        "\n# Include tests from viame-build\nsubdirs(\"${VIAME_TEST_DIR}\")\n")
    endif()
  else()
    # Create the file with the subdirs directive
    file(WRITE "${CTEST_FILE}"
      "# CMake generated Testfile for VIAME\nsubdirs(\"${VIAME_TEST_DIR}\")\n")
  endif()
endif()
