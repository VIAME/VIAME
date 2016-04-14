# sprokit External Project
#
# Required symbols are:
#   KWIVER_BUILD_PREFIX - where packages are built
#   KWIVER_BUILD_INSTALL_PREFIX - directory install target
#   KWIVER_PACKAGES_DIR - location of git submodule packages
#   KWIVER_ARGS_COMMON -
#   KWIVER_ENABLE_DOC - selection to build docs

# Produced symbols are:
#   KWIVER_ARGS_sprokit -
#

ExternalProject_Add( sprokit_proj
  PREFIX ${KWIVER_BUILD_PREFIX}
  SOURCE_DIR ${KWIVER_PACKAGES_DIR}/sprokit
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${KWIVER_ARGS_COMMON}
    -DSPROKIT_ENABLE_PYTHON:BOOL=${KWIVER_ENABLE_PYTHON}
    -DSPROKIT_ENABLE_DOCUMENTATION:BOOL=${KWIVER_ENABLE_DOCS}
    -DPYTHON_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}
    -DPYTHON_INCLUDE_DIR:PATH=${PYTHON_INCLUDE_DIR}
    -DPYTHON_LIBRARY:FILEPATH=${PYTHON_LIBRARY}
    -Ddoxy_documentation_output_path:STRING=${KWIVER_DOC_OUTPUT_DIR}
	-DSPROKIT_ENABLE_TESTING:BOOL=TRUE
  INSTALL_DIR ${KWIVER_BUILD_INSTALL_PREFIX}
  )

ExternalProject_Add_Step( sprokit_proj forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${KWIVER_BUILD_PREFIX}/src/sprokit-stamp/sprokit-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )

set( KWIVER_ARGS_sprokit
  -Dsprokit_DIR:PATH=${KWIVER_BUILD_INSTALL_PREFIX}/lib/cmake
  )

# symbols needed by sprokit CMake macros
set( sprokit_source_dir  ${KWIVER_PACKAGES_DIR}/sprokit )
set( sprokit_output_dir  ${KWIVER_BUILD_INSTALL_PREFIX} )
