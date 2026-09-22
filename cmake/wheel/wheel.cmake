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

# The CUDA variant, read from the toolkit this build used. It decides the
# nvidia wheel layout the RUNPATH targets, which requirements-cu* is layered
# on, and whether the wheel carries a local version. cu13 is the default and
# unmarked; cu12 is `viame+cu12`, for the Pascal and Volta hardware CUDA 13
# dropped. PyPI refuses local versions, so a variant needs its own index.
if( VIAME_ENABLE_CUDA AND CMAKE_CUDA_COMPILER AND NOT DEFINED VIAME_WHEEL_CUDA_MAJOR )
  if( CUDA_VERSION_MAJOR )
    set( VIAME_WHEEL_CUDA_MAJOR "${CUDA_VERSION_MAJOR}" )
  elseif( CUDAToolkit_VERSION_MAJOR )
    set( VIAME_WHEEL_CUDA_MAJOR "${CUDAToolkit_VERSION_MAJOR}" )
  endif()
endif()

set( _wheel_variant_args )
set( _wheel_variant_requires )
if( VIAME_WHEEL_CUDA_MAJOR )
  set( _variant_file
       "${VIAME_WHEEL_DIR}/requirements-cu${VIAME_WHEEL_CUDA_MAJOR}.txt" )
  if( NOT EXISTS "${_variant_file}" )
    message( FATAL_ERROR
      "No wheel requirements for CUDA ${VIAME_WHEEL_CUDA_MAJOR}. Add "
      "cmake/wheel/requirements-cu${VIAME_WHEEL_CUDA_MAJOR}.txt and a layout "
      "in CUDA_WHEEL_DIRS in build_wheel.py." )
  endif()
  list( APPEND _wheel_variant_args --cuda-major "${VIAME_WHEEL_CUDA_MAJOR}" )
  list( APPEND _wheel_variant_requires --requires-from "${_variant_file}" )
  if( NOT VIAME_WHEEL_CUDA_MAJOR EQUAL 13 )
    list( APPEND _wheel_variant_args
          --local-version "cu${VIAME_WHEEL_CUDA_MAJOR}" )
  endif()
  message( STATUS "  wheel: CUDA ${VIAME_WHEEL_CUDA_MAJOR} variant" )
endif()

# The interpreter the extension modules were built against decides the wheel's
# tag, so ask the build's python rather than whatever is first on PATH.
if( DEFINED PYTHON_EXECUTABLE )
  set( _viame_wheel_python "${PYTHON_EXECUTABLE}" )
elseif( DEFINED Python_EXECUTABLE )
  set( _viame_wheel_python "${Python_EXECUTABLE}" )
else()
  set( _viame_wheel_python "python3" )
endif()

# The default configs are selected from the install rather than listed by
# hand: which pipelines need no model is a property of the configs, and a
# hand-kept list would drift from them. Nothing from `configs/add-ons/` is
# ever selected -- add-on packs are model distributions, fetched at runtime.
set( VIAME_WHEEL_DEFAULT_CONFIGS
     "${VIAME_WHEEL_OUTPUT_DIR}/default-configs.txt" )

add_custom_target( wheel
  COMMAND "${_viame_wheel_python}"
          "${VIAME_WHEEL_DIR}/select_default_configs.py"
          --prefix "${CMAKE_INSTALL_PREFIX}"
          --manifest "${CMAKE_BINARY_DIR}/install_manifest.txt"
          --output "${VIAME_WHEEL_DEFAULT_CONFIGS}"
  COMMAND "${_viame_wheel_python}"
          "${VIAME_WHEEL_DIR}/build_wheel.py"
          --prefix     "${CMAKE_INSTALL_PREFIX}"
          --contents   "${VIAME_WHEEL_DIR}/contents.txt"
          --contents   "${VIAME_WHEEL_DEFAULT_CONFIGS}"
          --output-dir "${VIAME_WHEEL_OUTPUT_DIR}"
          --version    "${VIAME_WHEEL_VERSION}"
          --manifest   "${CMAKE_BINARY_DIR}/install_manifest.txt"
          --top-level  viame
          --top-level  kwiver
          --requires-from "${VIAME_WHEEL_DIR}/requirements.txt"
          ${_wheel_variant_requires}
          ${_wheel_variant_args}
          --license-file "${CMAKE_SOURCE_DIR}/LICENSE.txt"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  COMMENT "Packing ${CMAKE_INSTALL_PREFIX} into a wheel"
  VERBATIM
  USES_TERMINAL
  )
