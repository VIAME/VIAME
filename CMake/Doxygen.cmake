# Doxygen functions for the KWIVER project


find_package(Doxygen)

cmake_dependent_option(KWIVER_BUILD_DOC "Build documentation" OFF
  DOXYGEN_FOUND OFF)

if (KWIVER_BUILD_DOC)
  set(KWIVER_DOC_OUTPUT_DIR "${KWIVER_BUILD_INSTALL_PREFIX}/doc"
    CACHE PATH  "Location for documentation" )
endif()

if (DOXYGEN_FOUND)
  add_custom_target(doxygen
      COMMAND "${DOXYGEN_EXECUTABLE}" "${KWIVER_DOC_OUTPUT_DIR}/Doxyfile"
      WORKING_DIRECTORY "${KWIVER_DOC_OUTPUT_DIR}"
      ALL)
endif ()

## TODO
# need Doxyfile.in
# configure Doxyfile
