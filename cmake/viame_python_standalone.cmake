###
# A pinned CPython from python-build-standalone, in the install
#
# `VIAME_PYTHON_STANDALONE` (lite-build-system.md section 7): download a
# relocatable CPython build into `<install>/python` at configure time, and
# build the bindings and install the python dependencies against it. It is
# how a release that cannot assume a python on the target machine carries
# one, without compiling CPython -- `VIAME_BUILD_PYTHON_FROM_SOURCE` does
# that, on Linux only, and refuses on Windows.
#
# https://github.com/astral-sh/python-build-standalone publishes these builds;
# `install_only` archives unpack to a `python/` directory that runs from
# wherever it is put.
#
# Pinned: a release, a version, and the SHA256 of each platform's archive,
# from that release's SHA256SUMS. A different archive is a different pin.
##

set( _standalone_release 20260901 )
set( _standalone_version 3.12.14 )

if( WIN32 )
  set( _standalone_triple "x86_64-pc-windows-msvc" )
  set( _standalone_sha256 "e90c1b6419da3bd812dd73bb3de40287a21abf153438147639ec5e20375ea93f" )
elseif( APPLE )
  if( CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64" )
    set( _standalone_triple "aarch64-apple-darwin" )
    set( _standalone_sha256 "3ee3ee547cedfeb7c2b16b2b7156039f7b470bb8f857e226fd3d2eb11db83c76" )
  else()
    set( _standalone_triple "x86_64-apple-darwin" )
    set( _standalone_sha256 "2e31b23f3f1319f707d0e620b48847a0046577541d357276821f9f1b5492e0ba" )
  endif()
else()
  if( CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" )
    set( _standalone_triple "aarch64-unknown-linux-gnu" )
    set( _standalone_sha256 "b61b856c3e1a4fc65b8f6e6b0495ef975dd0924f90c59f3ea61b38a079173b84" )
  else()
    set( _standalone_triple "x86_64-unknown-linux-gnu" )
    set( _standalone_sha256 "936c246dfdbbfa7cb22dd01814a21f582a892689fae96b06071a5e433baffa22" )
  endif()
endif()

set( _standalone_archive
  "cpython-${_standalone_version}+${_standalone_release}-${_standalone_triple}-install_only.tar.gz" )
set( _standalone_url
  "https://github.com/astral-sh/python-build-standalone/releases/download/${_standalone_release}/${_standalone_archive}" )
set( _standalone_download "${VIAME_DOWNLOAD_DIR}/${_standalone_archive}" )
set( VIAME_PYTHON_STANDALONE_ROOT "${VIAME_BUILD_INSTALL_PREFIX}/python"
  CACHE INTERNAL "Where the standalone python is unpacked" )

# Download, unless the archive is already here and is the one pinned
set( _standalone_have FALSE )
if( EXISTS "${_standalone_download}" )
  file( SHA256 "${_standalone_download}" _standalone_existing )
  if( _standalone_existing STREQUAL _standalone_sha256 )
    set( _standalone_have TRUE )
  endif()
endif()

if( NOT _standalone_have )
  message( STATUS "Downloading python-build-standalone ${_standalone_version} (${_standalone_triple})" )
  file( MAKE_DIRECTORY "${VIAME_DOWNLOAD_DIR}" )
  file( DOWNLOAD "${_standalone_url}" "${_standalone_download}"
    EXPECTED_HASH SHA256=${_standalone_sha256}
    STATUS _standalone_status )
  list( GET _standalone_status 0 _standalone_code )
  if( NOT _standalone_code EQUAL 0 )
    list( GET _standalone_status 1 _standalone_message )
    file( REMOVE "${_standalone_download}" )
    message( FATAL_ERROR "Could not download ${_standalone_url}: ${_standalone_message}" )
  endif()
endif()

# Unpack once per archive, as the add-on packs are: a stamp beside the
# directory records the archive it holds.
set( _standalone_stamp "${VIAME_PYTHON_STANDALONE_ROOT}.sha256" )
set( _standalone_unpacked "" )
if( EXISTS "${_standalone_stamp}" AND IS_DIRECTORY "${VIAME_PYTHON_STANDALONE_ROOT}" )
  file( READ "${_standalone_stamp}" _standalone_unpacked )
  string( STRIP "${_standalone_unpacked}" _standalone_unpacked )
endif()

if( NOT _standalone_unpacked STREQUAL _standalone_sha256 )
  message( STATUS "Unpacking python-build-standalone into ${VIAME_PYTHON_STANDALONE_ROOT}" )
  file( REMOVE_RECURSE "${VIAME_PYTHON_STANDALONE_ROOT}" )
  file( REMOVE "${_standalone_stamp}" )
  file( MAKE_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}" )
  file( ARCHIVE_EXTRACT INPUT "${_standalone_download}" DESTINATION "${VIAME_BUILD_INSTALL_PREFIX}" )
  file( WRITE "${_standalone_stamp}" "${_standalone_sha256}\n" )
endif()

if( WIN32 )
  set( _standalone_python "${VIAME_PYTHON_STANDALONE_ROOT}/python.exe" )
else()
  set( _standalone_python "${VIAME_PYTHON_STANDALONE_ROOT}/bin/python3" )
endif()

if( NOT EXISTS "${_standalone_python}" )
  message( FATAL_ERROR "python-build-standalone unpacked, but ${_standalone_python} is not there" )
endif()

# FindPython takes this interpreter and nothing else
set( Python_EXECUTABLE "${_standalone_python}" CACHE FILEPATH "python-build-standalone" FORCE )
set( Python_ROOT_DIR "${VIAME_PYTHON_STANDALONE_ROOT}" )
set( Python_FIND_STRATEGY LOCATION )
set( Python_FIND_REGISTRY NEVER )
set( Python_FIND_FRAMEWORK NEVER )
set( Python_FIND_VIRTUALENV STANDARD )
set( VIAME_PYTHON_STANDALONE_VERSION "${_standalone_version}" )
