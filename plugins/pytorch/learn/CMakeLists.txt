
kwiver_create_python_init( pytorch/learn )

# -----------------------------------------------------------------------------
# Install pure python modules
# -----------------------------------------------------------------------------

install( DIRECTORY cutler tokencut mmdet
  DESTINATION ${VIAME_PYTHON_INSTALL}/viame/pytorch/learn
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

# -----------------------------------------------------------------------------
# Build and install vendored dependencies (pydensecrf, panopticapi)
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

if( Eigen3_INCLUDE_DIR )
  list( APPEND LEARN_ENV_VARS "EIGEN_INCLUDE_DIR=${Eigen3_INCLUDE_DIR}" )
endif()

# We use a custom target to ensure these are built during the build phase
# We use 'pip install --user' or 'pip install -e' depending on configuration,
# effectively mirroring what add_project_learn.cmake did.

if( VIAME_PYTHON_SYMLINK )
  set( LEARN_PIP_CMD
    ${CMAKE_COMMAND} -E env "${LEARN_ENV_VARS}"
    ${Python_EXECUTABLE} -m pip install --user -e . )
else()
  set( LEARN_PIP_CMD
    ${CMAKE_COMMAND} -E env "${LEARN_ENV_VARS}"
    ${Python_EXECUTABLE} -m pip install --user . )
endif()

add_custom_target( build_learn_deps ALL
  COMMAND cd ${PYDENSECRF_DIR} && ${LEARN_PIP_CMD}
  COMMAND cd ${PANOPTICAPI_DIR} && ${LEARN_PIP_CMD}
  COMMENT "Building and installing vendored learn dependencies"
)
