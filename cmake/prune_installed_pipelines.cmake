# Runs at install time, before the pipeline files are installed.
#
# An install never removes a file, so a pipeline that was renamed, deleted or
# belongs to an add-on since switched off would stay in configs/pipelines.
# Remove what the previous install put there and let the install rules put
# back what the current configuration produces. Only files the build itself
# installed go: those in the previous install manifest, and the add-on files
# configs/add-ons copies (which no manifest records, so it lists them in
# .build-installed). Add-ons installed by the add-ons tool are left alone.
#
# Expects CMAKE_INSTALL_PREFIX and VIAME_BUILD_MANIFEST.

set( _pipelines "${CMAKE_INSTALL_PREFIX}/configs/pipelines" )
set( _stale "" )

if( EXISTS "${VIAME_BUILD_MANIFEST}" )
  file( STRINGS "${VIAME_BUILD_MANIFEST}" _manifest )
  foreach( _file IN LISTS _manifest )
    if( _file MATCHES "/configs/pipelines/.*\\.(pipe|conf)$" AND EXISTS "${_file}" )
      list( APPEND _stale "${_file}" )
    endif()
  endforeach()
endif()

set( _addon_list "${_pipelines}/.build-installed" )
if( EXISTS "${_addon_list}" )
  file( STRINGS "${_addon_list}" _addon_files )
  foreach( _rel IN LISTS _addon_files )
    if( _rel MATCHES "\\.(pipe|conf)$" AND EXISTS "${_pipelines}/${_rel}" )
      list( APPEND _stale "${_pipelines}/${_rel}" )
    endif()
  endforeach()
  file( REMOVE "${_addon_list}" )
endif()

list( REMOVE_DUPLICATES _stale )
list( LENGTH _stale _count )
if( _count GREATER 0 )
  message( STATUS "Clearing ${_count} previously installed pipeline file(s)" )
  file( REMOVE ${_stale} )
endif()
