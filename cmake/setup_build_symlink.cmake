# setup_build_symlink.cmake
# Creates a symlink from SOURCE_DIR/build to BUILD_DIR
# This redirects build artifacts from source tree to build tree
#
# Required variables:
#   SOURCE_DIR - Source directory containing the package
#   BUILD_DIR  - Build directory where artifacts should go

if(NOT SOURCE_DIR OR NOT BUILD_DIR)
  message(FATAL_ERROR "SOURCE_DIR and BUILD_DIR must be defined")
endif()

set(LINK_PATH "${SOURCE_DIR}/build")

# Remove existing build directory or symlink
if(EXISTS "${LINK_PATH}" OR IS_SYMLINK "${LINK_PATH}")
  file(REMOVE_RECURSE "${LINK_PATH}")
endif()

# Create the build directory if it doesn't exist
file(MAKE_DIRECTORY "${BUILD_DIR}")

# Create symlink
file(CREATE_LINK "${BUILD_DIR}" "${LINK_PATH}" SYMBOLIC)
