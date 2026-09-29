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

# Which CPython series the install carries.
#
# The extension modules use the full CPython API and link a particular
# python3X, so there is no abi3 wheel: a wheel per interpreter is the only
# option, and a wheel matrix is this setting moved through its values. See
# cmake/wheel/build_matrix.sh and .ps1, which drive it.
set( VIAME_PYTHON_STANDALONE_VERSION "3.12" CACHE STRING
     "CPython series for the standalone interpreter" )
set_property( CACHE VIAME_PYTHON_STANDALONE_VERSION
              PROPERTY STRINGS "3.10" "3.11" "3.12" "3.13" "3.14" )

# Pinned for the same reason the interpreter is. See the setuptools block
# further down for why it has to be installed at all.
set( VIAME_STANDALONE_SETUPTOOLS "84.0.0" CACHE STRING
     "setuptools version installed into the standalone python" )

# The pin table: one patch version per series, and the SHA256 of each
# platform's archive, all taken from that release's SHA256SUMS. Nothing here
# is written from memory -- docs/wheels.md says how to regenerate it when the
# release moves, and a different archive is a different pin.
# BEGIN generated pins -- cmake/wheel/update_python_pins.py

set( _standalone_patch_3_10 "3.10.21" )
set( _standalone_sha_3_10_x86_64_pc_windows_msvc
     "b3edb1dacad300c7117ce4d6df97cd2e411afc0d173db43c224f904deb6253b5" )  # windows x86_64
set( _standalone_sha_3_10_aarch64_apple_darwin
     "cee232aabfb6790eec78f3cca935caeb7bd4eedca4dcb0a10dbcdb4302320b38" )  # macos arm64
set( _standalone_sha_3_10_x86_64_apple_darwin
     "e22c8cd258c12d264261104ae8ea5ce1b8e2c5efde074e36f01b7c299c17e221" )  # macos x86_64
set( _standalone_sha_3_10_aarch64_unknown_linux_gnu
     "bd9ebd39612e1ae9388983c4b89cad2c5b95a5b4eccd0f26288a4ab6a91dd50a" )  # linux aarch64
set( _standalone_sha_3_10_x86_64_unknown_linux_gnu
     "73cc92db5e6fb07ba611dca3709956d57cc2ed74ebcc01515b550ea89dfe9cba" )  # linux x86_64

set( _standalone_patch_3_11 "3.11.16" )
set( _standalone_sha_3_11_x86_64_pc_windows_msvc
     "6be524fa6752af802146a4adc7d098565425b0b1c166e19a5a7a4c8cccb86bf6" )  # windows x86_64
set( _standalone_sha_3_11_aarch64_apple_darwin
     "50424fa409e8ae84b82a3052522f64695b47dff2158b70bb7358e0ebd6c085c9" )  # macos arm64
set( _standalone_sha_3_11_x86_64_apple_darwin
     "167cc15cf4eeb72944a67bbd2f7120c45fded17d5043d5db64b3144d7adc30ae" )  # macos x86_64
set( _standalone_sha_3_11_aarch64_unknown_linux_gnu
     "c1cca4af741e33d47871b298842e6c3272cd9e8f57daf8085359fcbfe2d1a5aa" )  # linux aarch64
set( _standalone_sha_3_11_x86_64_unknown_linux_gnu
     "faa0758583a63f14c5eee516af82738403b59c13edda6fc0a21d953febd89eed" )  # linux x86_64

set( _standalone_patch_3_12 "3.12.14" )
set( _standalone_sha_3_12_x86_64_pc_windows_msvc
     "e90c1b6419da3bd812dd73bb3de40287a21abf153438147639ec5e20375ea93f" )  # windows x86_64
set( _standalone_sha_3_12_aarch64_apple_darwin
     "3ee3ee547cedfeb7c2b16b2b7156039f7b470bb8f857e226fd3d2eb11db83c76" )  # macos arm64
set( _standalone_sha_3_12_x86_64_apple_darwin
     "2e31b23f3f1319f707d0e620b48847a0046577541d357276821f9f1b5492e0ba" )  # macos x86_64
set( _standalone_sha_3_12_aarch64_unknown_linux_gnu
     "b61b856c3e1a4fc65b8f6e6b0495ef975dd0924f90c59f3ea61b38a079173b84" )  # linux aarch64
set( _standalone_sha_3_12_x86_64_unknown_linux_gnu
     "936c246dfdbbfa7cb22dd01814a21f582a892689fae96b06071a5e433baffa22" )  # linux x86_64

set( _standalone_patch_3_13 "3.13.15" )
set( _standalone_sha_3_13_x86_64_pc_windows_msvc
     "9bcc038a0bf180612ed56dec93d4977d035e80b8d9320ef51a38c287baf134b7" )  # windows x86_64
set( _standalone_sha_3_13_aarch64_apple_darwin
     "b9054a9d3d54f4cb5573d44907fddb29874b08909bde73f29f2868cf872223ee" )  # macos arm64
set( _standalone_sha_3_13_x86_64_apple_darwin
     "49f0d97f506b855eed60b74a8ac138595c5b39799a6aa5e0d7ca8abe1019a4d4" )  # macos x86_64
set( _standalone_sha_3_13_aarch64_unknown_linux_gnu
     "76ed18125286d7dc96ce24023d1e319dbd55a89a767102411b1ea23846113f69" )  # linux aarch64
set( _standalone_sha_3_13_x86_64_unknown_linux_gnu
     "0651dd7157d3debf769e15a52c1de9de7fbcdc36ba72faf79fde3c44f14d9461" )  # linux x86_64

set( _standalone_patch_3_14 "3.14.7" )
set( _standalone_sha_3_14_x86_64_pc_windows_msvc
     "5d9242012dded591d723a3a572dda265173ad58d8a3e6fdbc6dac8f94f36c80a" )  # windows x86_64
set( _standalone_sha_3_14_aarch64_apple_darwin
     "30daa970c7d223530120f1693cd3c6fa4c0c0d31ef158710b0dd77f286a5b23e" )  # macos arm64
set( _standalone_sha_3_14_x86_64_apple_darwin
     "dd8841a2e8ef94bd1a02b52f92843120942140f112145d4e0199abab56f120b1" )  # macos x86_64
set( _standalone_sha_3_14_aarch64_unknown_linux_gnu
     "30f1cc489be654477d895b441e196bb080738bf0456da82080ad4ab66a22d80f" )  # linux aarch64
set( _standalone_sha_3_14_x86_64_unknown_linux_gnu
     "0ab3305457051cd3e7c031857e005f1bda17c218a1990567dacaaac6dd1d14f0" )  # linux x86_64

# END generated pins

# A series ("3.12") or the full patch version it resolves to; only the first
# two components select.
string( REGEX MATCH "^[0-9]+\.[0-9]+" _standalone_series
        "${VIAME_PYTHON_STANDALONE_VERSION}" )
string( REPLACE "." "_" _standalone_key "${_standalone_series}" )

if( NOT DEFINED _standalone_patch_${_standalone_key} )
  message( FATAL_ERROR
    "VIAME_PYTHON_STANDALONE_VERSION is "
    "'${VIAME_PYTHON_STANDALONE_VERSION}'; the pins here cover 3.10, 3.11, "
    "3.12, 3.13 and 3.14 of python-build-standalone ${_standalone_release}." )
endif()

set( _standalone_version "${_standalone_patch_${_standalone_key}}" )

if( WIN32 )
  set( _standalone_triple "x86_64-pc-windows-msvc" )
elseif( APPLE )
  if( CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64" )
    set( _standalone_triple "aarch64-apple-darwin" )
  else()
    set( _standalone_triple "x86_64-apple-darwin" )
  endif()
else()
  if( CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" )
    set( _standalone_triple "aarch64-unknown-linux-gnu" )
  else()
    set( _standalone_triple "x86_64-unknown-linux-gnu" )
  endif()
endif()

string( REPLACE "-" "_" _standalone_triple_key "${_standalone_triple}" )
set( _standalone_sha256
     "${_standalone_sha_${_standalone_key}_${_standalone_triple_key}}" )

if( NOT _standalone_sha256 )
  message( FATAL_ERROR
    "No pinned python-build-standalone archive for ${_standalone_version} "
    "on ${_standalone_triple}." )
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

# CPython stopped bundling setuptools at 3.12, and python-build-standalone
# ships what CPython ships -- pip, and no setuptools. `cmake/packaging`'s
# `setup.py egg_info` needs it, and so does anything else that still reaches
# for distutils, so the interpreter VIAME carries gets it here rather than
# failing partway through the build. A system python usually already has it,
# which is why this only appears once the standalone one is used.
execute_process(
  COMMAND "${_standalone_python}" -c "import setuptools"
  RESULT_VARIABLE _standalone_has_setuptools
  OUTPUT_QUIET ERROR_QUIET )

if( NOT _standalone_has_setuptools EQUAL 0 )
  message( STATUS "Installing setuptools ${VIAME_STANDALONE_SETUPTOOLS} into the standalone python" )
  execute_process(
    COMMAND "${_standalone_python}" -m pip install
            --disable-pip-version-check --no-warn-script-location
            "setuptools==${VIAME_STANDALONE_SETUPTOOLS}"
    RESULT_VARIABLE _standalone_pip
    OUTPUT_VARIABLE _standalone_pip_output
    ERROR_VARIABLE  _standalone_pip_output )

  if( NOT _standalone_pip EQUAL 0 )
    message( FATAL_ERROR
      "Could not install setuptools into ${_standalone_python}: "
      "${_standalone_pip_output}" )
  endif()
endif()

# FindPython takes this interpreter and nothing else
set( Python_EXECUTABLE "${_standalone_python}" CACHE FILEPATH "python-build-standalone" FORCE )
set( Python_ROOT_DIR "${VIAME_PYTHON_STANDALONE_ROOT}" )
set( Python_FIND_STRATEGY LOCATION )
set( Python_FIND_REGISTRY NEVER )
set( Python_FIND_FRAMEWORK NEVER )
set( Python_FIND_VIRTUALENV STANDARD )
# The cache entry holds the series the user picked; this shadows it with the
# patch version actually unpacked, which is what find_package( Python EXACT )
# in viame_dependencies.cmake is given.
set( VIAME_PYTHON_STANDALONE_VERSION "${_standalone_version}" )
