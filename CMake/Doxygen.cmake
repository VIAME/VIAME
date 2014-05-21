# Doxygen functions for the KWIVER project


find_package(Doxygen)

cmake_dependent_option(KWIVER_BUILD_DOC "Build documentation" OFF
  DOXYGEN_FOUND OFF)

set(KWIVER_DOC_OUTPUT_DIR  "${KWIVER_BUILD_INSTALL_PREFIX}/doc")

if (DOXYGEN_FOUND)
  add_custom_target(doxygen)
endif ()

# need Doxyfile.in
# configure Doxyfile
