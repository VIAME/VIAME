# Download an archive and unpack it into a directory.
#
# What `ExternalProject_Add` did for the handful of VIAME dependencies that
# are a URL and nothing else -- no configure, no build, just fetch and copy.
# ExternalProject brings a whole build-step state machine along for that,
# and only works inside a superbuild; this is the same thing as a script.
#
#   URL           what to fetch
#   DESTINATION   where its contents end up
#   DOWNLOAD_DIR  scratch space, kept so a re-run does not re-fetch
#   EXPECTED_MD5  optional, and checked when given -- these are release
#                 binaries fetched over the network, so an archive that is
#                 not the one the build was written against should stop the
#                 build rather than be unpacked into the install

cmake_minimum_required( VERSION 3.16 )

foreach( _required URL DESTINATION DOWNLOAD_DIR )
  if( NOT ${_required} )
    message( FATAL_ERROR "viame_fetch_archive: ${_required} is required" )
  endif()
endforeach()

get_filename_component( _name "${URL}" NAME )
set( _archive "${DOWNLOAD_DIR}/${_name}" )

file( MAKE_DIRECTORY "${DOWNLOAD_DIR}" )

if( EXISTS "${_archive}" AND EXPECTED_MD5 )
  file( MD5 "${_archive}" _have )
  if( NOT _have STREQUAL EXPECTED_MD5 )
    message( STATUS "${_name} does not match its checksum; fetching again" )
    file( REMOVE "${_archive}" )
  endif()
endif()

if( NOT EXISTS "${_archive}" )
  message( STATUS "Fetching ${URL}" )

  if( EXPECTED_MD5 )
    file( DOWNLOAD "${URL}" "${_archive}"
          STATUS _status
          EXPECTED_MD5 ${EXPECTED_MD5}
          SHOW_PROGRESS )
  else()
    file( DOWNLOAD "${URL}" "${_archive}"
          STATUS _status
          SHOW_PROGRESS )
  endif()

  list( GET _status 0 _code )
  if( NOT _code EQUAL 0 )
    list( GET _status 1 _message )
    file( REMOVE "${_archive}" )
    message( FATAL_ERROR "Could not fetch ${URL}: ${_message}" )
  endif()
else()
  message( STATUS "${_name} is already downloaded" )
endif()

set( _unpacked "${DOWNLOAD_DIR}/unpacked" )

file( REMOVE_RECURSE "${_unpacked}" )
file( MAKE_DIRECTORY "${_unpacked}" )
file( ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${_unpacked}" )

# These archives carry one top-level directory named after the release. What
# the caller wants is its contents, not a directory whose name has a version
# in it.
file( GLOB _entries "${_unpacked}/*" )
list( LENGTH _entries _count )

if( _count EQUAL 1 AND IS_DIRECTORY "${_entries}" )
  set( _root "${_entries}" )
else()
  set( _root "${_unpacked}" )
endif()

file( MAKE_DIRECTORY "${DESTINATION}" )
file( COPY "${_root}/" DESTINATION "${DESTINATION}" )

message( STATUS "Unpacked ${_name} into ${DESTINATION}" )
