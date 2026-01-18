# Learn package for VIAME PyTorch

# Helper function to recursively add all Python modules in a directory
function(add_python_modules_recursive base_path install_prefix)
  file(GLOB_RECURSE _python_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
       "${base_path}/*.py")

  foreach(_py_file ${_python_files})
    # Get the directory and filename
    get_filename_component(_py_dir ${_py_file} DIRECTORY)
    get_filename_component(_py_name ${_py_file} NAME_WE)

    # Build the module path
    if(_py_dir)
      set(_mod_path "${install_prefix}/${_py_dir}")
    else()
      set(_mod_path "${install_prefix}")
    endif()

    kwiver_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/${_py_file}
      ${_mod_path}
      ${_py_name}
    )
  endforeach()
endfunction()

# Add all Python modules from learn directory (cutler, tokencut, mmdet)
# The base_path is used for GLOB_RECURSE which returns paths like "cutler/__init__.py",
# so we use "pytorch/learn" as install_prefix (not "pytorch/learn/cutler")
add_python_modules_recursive( "cutler" "pytorch/learn" )
add_python_modules_recursive( "tokencut" "pytorch/learn" )
add_python_modules_recursive( "mmdet" "pytorch/learn" )

# Also add the root __init__.py if it's not empty
kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  pytorch/learn
  __init__ )

# -----------------------------------------------------------------------------
# Build and install vendored dependencies (pydensecrf, panopticapi)
# These have C++ extensions and need to be built via pip
# -----------------------------------------------------------------------------

set( LEARN_DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
set( PYDENSECRF_DIR ${LEARN_DEPS_DIR}/pydensecrf )
set( PANOPTICAPI_DIR ${LEARN_DEPS_DIR}/panopticapi )

set( LEARN_BUILD_DIR ${VIAME_BUILD_PREFIX}/src/learn-build )
file( MAKE_DIRECTORY ${LEARN_BUILD_DIR} )

set( LEARN_ENV_VARS ${PYTHON_DEP_ENV_VARS} )

if( VIAME_ENABLE_CUDA )
  list( APPEND LEARN_ENV_VARS "USE_CUDA=1" )
  list( APPEND LEARN_ENV_VARS "CUDA_VISIBLE_DEVICES=0" )
  list( APPEND LEARN_ENV_VARS "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}" )
  list( APPEND LEARN_ENV_VARS "TORCH_CUDA_ARCH_LIST=${CUDA_ARCHITECTURES}" )
  list( APPEND LEARN_ENV_VARS "TORCH_NVCC_FLAGS=-Xfatbin -compress-all" )
else()
  list( APPEND LEARN_ENV_VARS "USE_CUDA=0" )
endif()

# Find Eigen include directory - try multiple approaches
if( NOT EIGEN3_INCLUDE_DIR )
  # Try to find via find_package
  find_package( Eigen3 QUIET )
  if( EIGEN3_FOUND )
    set( EIGEN3_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}" )
  elseif( EXISTS "${VIAME_BUILD_INSTALL_PREFIX}/include/eigen3" )
    # Fall back to VIAME install location
    set( EIGEN3_INCLUDE_DIR "${VIAME_BUILD_INSTALL_PREFIX}/include/eigen3" )
  endif()
endif()

if( EIGEN3_INCLUDE_DIR )
  list( APPEND LEARN_ENV_VARS "EIGEN_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR}" )
endif()

# Pass Python include directory for building C extensions
# On Windows with MSVC, use INCLUDE environment variable
if( Python_INCLUDE_DIR )
  list( APPEND LEARN_ENV_VARS "CFLAGS=-I${Python_INCLUDE_DIR}" )
  list( APPEND LEARN_ENV_VARS "CXXFLAGS=-I${Python_INCLUDE_DIR}" )
  if( WIN32 )
    list( APPEND LEARN_ENV_VARS "INCLUDE=${Python_INCLUDE_DIR}" )
  endif()
elseif( Python_INCLUDE_DIRS )
  list( APPEND LEARN_ENV_VARS "CFLAGS=-I${Python_INCLUDE_DIRS}" )
  list( APPEND LEARN_ENV_VARS "CXXFLAGS=-I${Python_INCLUDE_DIRS}" )
  if( WIN32 )
    list( APPEND LEARN_ENV_VARS "INCLUDE=${Python_INCLUDE_DIRS}" )
  endif()
endif()

# Install vendored dependencies to VIAME's Python install location
set( LEARN_INSTALL_TARGET "${VIAME_PYTHON_INSTALL}/site-packages" )

# Build wheels in the build directory, then install them
# This avoids creating build artifacts in the source tree
set( PYDENSECRF_BUILD_DIR ${LEARN_BUILD_DIR}/pydensecrf )
set( PANOPTICAPI_BUILD_DIR ${LEARN_BUILD_DIR}/panopticapi )
file( MAKE_DIRECTORY ${PYDENSECRF_BUILD_DIR} )
file( MAKE_DIRECTORY ${PANOPTICAPI_BUILD_DIR} )

# Convert env vars to ----separated string for custom_build_python_dep.cmake
# This ensures <PS> placeholders are properly converted to platform path separators
string( REPLACE ";" "----" LEARN_ENV_VARS_STR "${LEARN_ENV_VARS}" )

if( VIAME_PYTHON_SYMLINK )
  # For symlink mode, use editable installs
  set( PYDENSECRF_PIP_CMD
    ${Python_EXECUTABLE} -m pip install --no-build-isolation --no-deps --target "${LEARN_INSTALL_TARGET}" -e . )
  set( PANOPTICAPI_PIP_CMD
    ${Python_EXECUTABLE} -m pip install --no-build-isolation --no-deps --target "${LEARN_INSTALL_TARGET}" -e . )

  string( REPLACE ";" "----" PYDENSECRF_PIP_CMD_STR "${PYDENSECRF_PIP_CMD}" )
  string( REPLACE ";" "----" PANOPTICAPI_PIP_CMD_STR "${PANOPTICAPI_PIP_CMD}" )

  set( PYDENSECRF_HASH_FILE ${LEARN_BUILD_DIR}/pydensecrf-hash.txt )
  set( PANOPTICAPI_HASH_FILE ${LEARN_BUILD_DIR}/panopticapi-hash.txt )

  set( PYDENSECRF_BUILD_SCRIPT
    ${CMAKE_COMMAND}
      -DLIB_NAME=pydensecrf
      -DLIB_SOURCE_DIR=${PYDENSECRF_DIR}
      -DHASH_FILE=${PYDENSECRF_HASH_FILE}
      -DPYTHON_BUILD_CMD=${PYDENSECRF_PIP_CMD_STR}
      "-DENV_VARS:STRING=${LEARN_ENV_VARS_STR}"
      -DWORKING_DIR:PATH=${PYDENSECRF_DIR}
      -P ${VIAME_CMAKE_DIR}/custom_build_python_dep.cmake )

  set( PANOPTICAPI_BUILD_SCRIPT
    ${CMAKE_COMMAND}
      -DLIB_NAME=panopticapi
      -DLIB_SOURCE_DIR=${PANOPTICAPI_DIR}
      -DHASH_FILE=${PANOPTICAPI_HASH_FILE}
      -DPYTHON_BUILD_CMD=${PANOPTICAPI_PIP_CMD_STR}
      "-DENV_VARS:STRING=${LEARN_ENV_VARS_STR}"
      -DWORKING_DIR:PATH=${PANOPTICAPI_DIR}
      -P ${VIAME_CMAKE_DIR}/custom_build_python_dep.cmake )

  add_custom_target( build_learn_deps ALL
    COMMAND ${PYDENSECRF_BUILD_SCRIPT}
    COMMAND ${PANOPTICAPI_BUILD_SCRIPT}
    COMMENT "Building and installing vendored learn dependencies (pydensecrf, panopticapi)"
  )
else()
  # Build wheels in the build directory, then install from there
  # Build and install combined for each package to handle <PS> properly
  set( PYDENSECRF_WHEEL_CMD
    ${Python_EXECUTABLE} -m pip wheel --no-build-isolation --no-deps -w ${PYDENSECRF_BUILD_DIR} ${PYDENSECRF_DIR} )
  set( PANOPTICAPI_WHEEL_CMD
    ${Python_EXECUTABLE} -m pip wheel --no-build-isolation --no-deps -w ${PANOPTICAPI_BUILD_DIR} ${PANOPTICAPI_DIR} )

  string( REPLACE ";" "----" PYDENSECRF_WHEEL_CMD_STR "${PYDENSECRF_WHEEL_CMD}" )
  string( REPLACE ";" "----" PANOPTICAPI_WHEEL_CMD_STR "${PANOPTICAPI_WHEEL_CMD}" )

  set( PYDENSECRF_HASH_FILE ${LEARN_BUILD_DIR}/pydensecrf-hash.txt )
  set( PANOPTICAPI_HASH_FILE ${LEARN_BUILD_DIR}/panopticapi-hash.txt )

  set( PYDENSECRF_BUILD_SCRIPT
    ${CMAKE_COMMAND}
      -DLIB_NAME=pydensecrf
      -DLIB_SOURCE_DIR=${PYDENSECRF_DIR}
      -DHASH_FILE=${PYDENSECRF_HASH_FILE}
      -DPYTHON_BUILD_CMD=${PYDENSECRF_WHEEL_CMD_STR}
      "-DENV_VARS:STRING=${LEARN_ENV_VARS_STR}"
      -DWORKING_DIR:PATH=${PYDENSECRF_DIR}
      -P ${VIAME_CMAKE_DIR}/custom_build_python_dep.cmake )

  set( PANOPTICAPI_BUILD_SCRIPT
    ${CMAKE_COMMAND}
      -DLIB_NAME=panopticapi
      -DLIB_SOURCE_DIR=${PANOPTICAPI_DIR}
      -DHASH_FILE=${PANOPTICAPI_HASH_FILE}
      -DPYTHON_BUILD_CMD=${PANOPTICAPI_WHEEL_CMD_STR}
      "-DENV_VARS:STRING=${LEARN_ENV_VARS_STR}"
      -DWORKING_DIR:PATH=${PANOPTICAPI_DIR}
      -P ${VIAME_CMAKE_DIR}/custom_build_python_dep.cmake )

  # For install commands, we need to set PYTHONHOME to avoid environment conflicts
  # We reuse the properly-converted LEARN_ENV_VARS which includes PYTHONHOME
  # but process them through a cmake script to handle <PS> properly
  set( PYDENSECRF_INSTALL_PIP_CMD
    ${Python_EXECUTABLE} -m pip install --no-deps --no-index --target "${LEARN_INSTALL_TARGET}" --find-links ${PYDENSECRF_BUILD_DIR} pydensecrf )
  set( PANOPTICAPI_INSTALL_PIP_CMD
    ${Python_EXECUTABLE} -m pip install --no-deps --no-index --target "${LEARN_INSTALL_TARGET}" --find-links ${PANOPTICAPI_BUILD_DIR} panopticapi )

  string( REPLACE ";" "----" PYDENSECRF_INSTALL_PIP_CMD_STR "${PYDENSECRF_INSTALL_PIP_CMD}" )
  string( REPLACE ";" "----" PANOPTICAPI_INSTALL_PIP_CMD_STR "${PANOPTICAPI_INSTALL_PIP_CMD}" )

  # Use same hash files (the install doesn't need separate hashing)
  set( PYDENSECRF_INSTALL_SCRIPT
    ${CMAKE_COMMAND}
      -DLIB_NAME=pydensecrf-install
      -DLIB_SOURCE_DIR=${PYDENSECRF_DIR}
      -DHASH_FILE=${LEARN_BUILD_DIR}/pydensecrf-install-hash.txt
      -DPYTHON_BUILD_CMD=${PYDENSECRF_INSTALL_PIP_CMD_STR}
      "-DENV_VARS:STRING=${LEARN_ENV_VARS_STR}"
      -DWORKING_DIR:PATH=${PYDENSECRF_BUILD_DIR}
      -P ${VIAME_CMAKE_DIR}/custom_build_python_dep.cmake )

  set( PANOPTICAPI_INSTALL_SCRIPT
    ${CMAKE_COMMAND}
      -DLIB_NAME=panopticapi-install
      -DLIB_SOURCE_DIR=${PANOPTICAPI_DIR}
      -DHASH_FILE=${LEARN_BUILD_DIR}/panopticapi-install-hash.txt
      -DPYTHON_BUILD_CMD=${PANOPTICAPI_INSTALL_PIP_CMD_STR}
      "-DENV_VARS:STRING=${LEARN_ENV_VARS_STR}"
      -DWORKING_DIR:PATH=${PANOPTICAPI_BUILD_DIR}
      -P ${VIAME_CMAKE_DIR}/custom_build_python_dep.cmake )

  add_custom_target( build_learn_deps ALL
    COMMAND ${PYDENSECRF_BUILD_SCRIPT}
    COMMAND ${PYDENSECRF_INSTALL_SCRIPT}
    COMMAND ${PANOPTICAPI_BUILD_SCRIPT}
    COMMAND ${PANOPTICAPI_INSTALL_SCRIPT}
    COMMENT "Building and installing vendored learn dependencies (pydensecrf, panopticapi)"
  )
endif()
