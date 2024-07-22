# PYMOTMETRICS External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} pymotmetrics )

set( PYMOTMETRICS_DEPENDS fletch python-deps )

if( VIAME_ENABLE_PYTHON )
  FormatPassdowns( "Python" VIAME_PYTHON_FLAGS )
endif()

if( VIAME_SYMLINK_PYTHON )
  set( PYMOTMETRICS_PIP_CMD
    pip install --user -e .[postgres] )
else()
  # This is only required for no symlink install without a -e with older
  # versions of pip, for never versions the above command works with no -e
  set( PYMOTMETRICS_PIP_CMD
    pip install --user file://${VIAME_PACKAGES_DIR}/pymotmetrics\#egg=pymotmetrics[postgres] )
endif()

if( WIN32 )
  #list( APPEND PYMOTMETRICS_DEPENDS postgres )
endif()

set( PYMOTMETRICS_PYTHON_INSTALL
  ${CMAKE_COMMAND} -E env "${PYTHON_DEP_ENV_VARS}"
    ${Python_EXECUTABLE} -m ${PYMOTMETRICS_PIP_CMD}
  )

ExternalProject_Add( pymotmetrics
  DEPENDS ${PYMOTMETRICS_DEPENDS}
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/pymotmetrics
  USES_TERMINAL_BUILD 1
  CMAKE_GENERATOR ${gen}
  CMAKE_CACHE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}
    ${VIAME_PYTHON_FLAGS}

  INSTALL_DIR ${VIAME_INSTALL_PREFIX}
  LIST_SEPARATOR "----"
  )

ExternalProject_Add_Step(pymotmetrics install_python
  WORKING_DIRECTORY ${VIAME_PACKAGES_DIR}/pymotmetrics
  COMMAND ${PYMOTMETRICS_PYTHON_INSTALL}
  COMMENT "Installing PYMOTMETRICS python files."
  DEPENDEES build
  )

if ( VIAME_FORCEBUILD )
ExternalProject_Add_Step(pymotmetrics forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${VIAME_BUILD_PREFIX}/src/pymotmetrics-stamp/pymotmetrics-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )
endif()

set(VIAME_ARGS_pymotmetrics
  -DPYMOTMETRICS_DIR:PATH=${VIAME_BUILD_PREFIX}/src/pymotmetrics-build
  )
