
# This file handles pydensecrf and panopticapi dependencies
# now vendored in plugins/pytorch/learn

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} learn_deps )

# Use local vendored directories
# Use local vendored directories
set( LEARN_DEPS_DIR ${VIAME_SOURCE_DIR}/plugins/pytorch/learn )

set( PYDENSECRF_DIR ${LEARN_DEPS_DIR}/pydensecrf )
set( PANOPTICAPI_DIR ${LEARN_DEPS_DIR}/panopticapi )
set( REMAX_DIR ${VIAME_SOURCE_DIR}/plugins/pytorch/remax )
set( REMAX_OPS_DIR ${REMAX_DIR}/model/ops )

set( LEARN_BUILD_DIR ${VIAME_BUILD_PREFIX}/src/learn-build )

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

# Setup python env vars and commands
set( LEARN_REQ_PIP_CMD
    ${CMAKE_COMMAND} -E env "${LEARN_ENV_VARS}"
    ${Python_EXECUTABLE} -m pip install --user )

if( VIAME_PYTHON_SYMLINK )
  set( LEARN_BUILD_CMD
    ${CMAKE_COMMAND} -E env "${LEARN_ENV_VARS}"
    ${Python_EXECUTABLE} -m pip install --user -e . )
  set( LEARN_INSTALL_CMD )
else()
  set( LEARN_BUILD_CMD
    ${CMAKE_COMMAND} -E env "${LEARN_ENV_VARS}"
    ${Python_EXECUTABLE} setup.py
      build --build-base=${LEARN_BUILD_DIR}
      bdist_wheel -d ${LEARN_BUILD_DIR} )
  set( LEARN_INSTALL_CMD
    ${CMAKE_COMMAND} -E env "${LEARN_ENV_VARS}"
    ${CMAKE_COMMAND} -DWHEEL_DIR=${LEARN_BUILD_DIR}
    -DPYTHON_EXECUTABLE=${Python_EXECUTABLE} -DPython_EXECUTABLE=${Python_EXECUTABLE}
    -P ${VIAME_CMAKE_DIR}/install_python_wheel.cmake )
endif()

# Install required dependencies
ExternalProject_Add( learn_deps
    DEPENDS python-deps detectron2 torchvideo
    PREFIX ${VIAME_BUILD_PREFIX}
    SOURCE_DIR ${VIAME_PACKAGES_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${LEARN_REQ_PIP_CMD} -r ${REMAX_DIR}/requirements.txt
      COMMAND cd ${PYDENSECRF_DIR} && ${LEARN_BUILD_CMD}
      COMMAND cd ${PANOPTICAPI_DIR} && ${LEARN_BUILD_CMD}
      COMMAND cd ${REMAX_OPS_DIR} && ${LEARN_BUILD_CMD}
    INSTALL_COMMAND ${LEARN_INSTALL_CMD}
    LIST_SEPARATOR "----"
    )
