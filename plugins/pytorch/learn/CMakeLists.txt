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

# Install vendored dependencies to VIAME's Python install location
set( LEARN_INSTALL_TARGET "${VIAME_PYTHON_INSTALL}/site-packages" )

if( VIAME_PYTHON_SYMLINK )
  set( LEARN_PIP_CMD
    ${CMAKE_COMMAND} -E env "${LEARN_ENV_VARS}"
    ${Python_EXECUTABLE} -m pip install --no-build-isolation --no-deps --target "${LEARN_INSTALL_TARGET}" -e . )
else()
  set( LEARN_PIP_CMD
    ${CMAKE_COMMAND} -E env "${LEARN_ENV_VARS}"
    ${Python_EXECUTABLE} -m pip install --no-build-isolation --no-deps --target "${LEARN_INSTALL_TARGET}" . )
endif()

add_custom_target( build_learn_deps ALL
  COMMAND cd ${PYDENSECRF_DIR} && ${LEARN_PIP_CMD}
  COMMAND cd ${PANOPTICAPI_DIR} && ${LEARN_PIP_CMD}
  COMMENT "Building and installing vendored learn dependencies (pydensecrf, panopticapi)"
)
