# vibrant External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} smqtk )

ExternalProject_Add( smqtk
  DEPENDS fletch
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/smqtk
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    -DSMQTK_BUILD_EXAMPLES:BOOL=OFF
    -DSMQTK_BUILD_LIBSVM:BOOL=OFF
    -DSMQTK_INSTALL_SETUP_SCRIPT:BOOL=OFF
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}

  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  )

ExternalProject_Add_Step(smqtk forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${VIAME_BUILD_PREFIX}/src/smqtk-stamp/smqtk-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )

set(VIAME_ARGS_smqtk
  -Dsmqtk_DIR:PATH=${VIAME_BUILD_PREFIX}/src/smqtk-build
  )
