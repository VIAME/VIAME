###
# The `wheel` target
#
# Packs an already-installed VIAME into a PEP 427 wheel. It is deliberately
# *not* part of `all`, and it does not build or install anything itself: a
# wheel is made from an install prefix, and making the target depend on the
# install would mean a `make wheel` that quietly reinstalls over a prefix the
# user may be running from.
#
#   make install
#   make wheel
#
# What goes in is `cmake/wheel/contents.txt`, which is this branch's half of
# the arrangement; `build_wheel.py` is shared with the other branches and
# knows nothing about either layout. See the header of each.
##

option( VIAME_ENABLE_WHEEL "Add the `wheel` target" ON )

if( NOT VIAME_ENABLE_WHEEL )
  return()
endif()

set( VIAME_WHEEL_DIR "${CMAKE_CURRENT_LIST_DIR}" )

# The wheel's version. `VIAME_VERSION` is the release number; a build that is
# not a tagged release says so, because an untagged wheel that claims to be
# 1.0.0 is the kind of thing that ends up installed somewhere and cannot be
# told apart from the real one.
if( NOT DEFINED VIAME_WHEEL_VERSION )
  if( VIAME_VERSION_RELEASE )
    set( VIAME_WHEEL_VERSION "${VIAME_VERSION}" )
  else()
    set( VIAME_WHEEL_VERSION "${VIAME_VERSION}.dev0" )
  endif()
endif()

set( VIAME_WHEEL_OUTPUT_DIR "${CMAKE_BINARY_DIR}/wheel"
     CACHE PATH "Where `make wheel` writes the .whl" )

# The interpreter the extension modules were built against decides the wheel's
# tag, so ask the build's python rather than whatever is first on PATH.
if( DEFINED PYTHON_EXECUTABLE )
  set( _viame_wheel_python "${PYTHON_EXECUTABLE}" )
elseif( DEFINED Python_EXECUTABLE )
  set( _viame_wheel_python "${Python_EXECUTABLE}" )
else()
  set( _viame_wheel_python "python3" )
endif()

add_custom_target( wheel
  COMMAND "${_viame_wheel_python}"
          "${VIAME_WHEEL_DIR}/build_wheel.py"
          --prefix     "${CMAKE_INSTALL_PREFIX}"
          --contents   "${VIAME_WHEEL_DIR}/contents.txt"
          --output-dir "${VIAME_WHEEL_OUTPUT_DIR}"
          --version    "${VIAME_WHEEL_VERSION}"
          --manifest   "${CMAKE_BINARY_DIR}/install_manifest.txt"
          --top-level  viame
          --top-level  kwiver
          --requires   "numpy>=1.13.0"
          --license-file "${CMAKE_SOURCE_DIR}/LICENSE.txt"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  COMMENT "Packing ${CMAKE_INSTALL_PREFIX} into a wheel"
  VERBATIM
  USES_TERMINAL
  )
