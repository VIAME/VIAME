# vivia External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} vivia )

if( APPLE )
  set( VIVIA_DISABLE_FIXUP OFF )
else()
  set( VIVIA_DISABLE_FIXUP ON )
endif()

ExternalProject_Add(vivia
  DEPENDS fletch kwiver
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/vivia
  USES_TERMINAL_BUILD 1
  CMAKE_GENERATOR ${gen}
  CMAKE_CACHE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_Boost}
    ${VIAME_ARGS_fletch}
    ${VIAME_ARGS_kwiver}
    ${VIAME_ARGS_libkml}
    ${VIAME_ARGS_PROJ4}
    ${VIAME_ARGS_VTK}
    ${VIAME_ARGS_VXL_INSTALL}
    ${VIAME_ARGS_Qt}

    # Required
    -DBUILD_SHARED_LIBS:BOOL=ON
    -DVISGUI_ENABLE_VIDTK:BOOL=OFF

    -DVISGUI_ENABLE_VIQUI:BOOL=${VIAME_ENABLE_SVM}
    -DVISGUI_ENABLE_VSPLAY:BOOL=OFF
    -DVISGUI_ENABLE_VPVIEW:BOOL=ON
    -DVISGUI_ENABLE_GDAL:BOOL=${VIAME_ENABLE_GDAL}

    -DVISGUI_ENABLE_FAKE_STREAM_SOURCE:BOOL=OFF
    -DVSPSS_ENABLE_FAKE_STREAM_SOURCE:BOOL=OFF
    -DVSPUI_ENABLE_CONTEXT_VIEW:BOOL=OFF
    -DVSPUI_ENABLE_CONTEXT_VIEWER:BOOL=OFF
    -DVSPUI_ENABLE_EVENT_CREATION_TOOL:BOOL=OFF
    -DVSPUI_ENABLE_EVENT_CREATION_TOOLS:BOOL=OFF
    -DVSPUI_ENABLE_KML_EXPORT:BOOL=OFF
    -DVSPUI_ENABLE_REPORT_GENERATOR:BOOL=OFF
    -DVISGUI_ENABLE_KWIVER:BOOL=${VIAME_ENABLE_KWIVER}
    -DVVQS_ENABLE_FAKE_BACKEND:BOOL=OFF
    -DVISGUI_DISABLE_FIXUP_BUNDLE:BOOL=${VIVIA_DISABLE_FIXUP}

    -DLIBJSON_INCLUDE_DIR:PATH=${VIAME_BUILD_INSTALL_PREFIX}/include/json

  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  )

if( VIAME_BUILD_FORCE_REBUILD )
  RemoveProjectCMakeStamp( vivia )
endif()

# vivia's find_package(geographiclib) loads geographiclib-targets.cmake, which
# declares imported targets referencing install/bin/<tool>.exe (CartConvert,
# ConicProj, GeodSolve, etc.). Fletch builds these, but on clean Windows
# builds only some end up in install/bin (cause not yet pinned down). Ensure
# all 11 tool executables are present immediately before vivia configures.
if( WIN32 )
  set( _GEO_TOOLS CartConvert ConicProj GeodesicProj GeoConvert GeodSolve
                  GeoidEval Gravity MagneticField Planimeter RhumbSolve
                  TransverseMercatorProj )
  set( _GEO_COPY_CMDS )
  foreach( _tool ${_GEO_TOOLS} )
    list( APPEND _GEO_COPY_CMDS
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${VIAME_BUILD_FLETCH_DIR}/build/src/GeographicLib-build/bin/Release/${_tool}.exe"
        "${VIAME_BUILD_INSTALL_PREFIX}/bin/${_tool}.exe" )
  endforeach()
  ExternalProject_Add_Step( vivia ensure_geographiclib_tools
    ${_GEO_COPY_CMDS}
    DEPENDEES patch
    DEPENDERS configure
    COMMENT "VIAME: ensuring GeographicLib tool executables are in install/bin"
  )
endif()

set(VIAME_ARGS_vivia
  -Dvivia_DIR:PATH=${VIAME_BUILD_PREFIX}/src/vivia-build
  )
