# VIVIA External Project
#
# Required symbols are:
#   KWIVER_BUILD_PREFIX - where packages are built
#   KWIVER_BUILD_INSTALL_PREFIX - directory install target
#   KWIVER_PACKAGES_DIR - location of git submodule packages
#   KWIVER_ARGS_COMMON -
#
# Produced symbols are:
#   KWIVER_ARGS_vivia -
#

ExternalProject_Add(vivia
  PREFIX ${KWIVER_BUILD_PREFIX}
  SOURCE_DIR ${KWIVER_PACKAGES_DIR}/vivia
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${KWIVER_ARGS_COMMON}
	${KWIVER_ARGS_VIVIA_APPS}
	${KWIVER_ARGS_VIVIA_DEPS}
	-DVISGUI_DISABLE_FIXUP_BUNDLE:BOOL=TRUE
  INSTALL_DIR ${KWIVER_BUILD_INSTALL_PREFIX}
  DEPENDS VXL vibrant
  )

ExternalProject_Add_Step(vivia forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${KWIVER_BUILD_PREFIX}/src/vivia-stamp/vivia-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )
