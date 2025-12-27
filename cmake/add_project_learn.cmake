
set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} learn )

set( LEARN_DIR ${VIAME_SOURCE_DIR}/packages/learn )
set( LEARN_DEPS_DIR ${LEARN_DIR}-deps )

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

if( NOT EXISTS "${LEARN_DEPS_DIR}" )
  file( MAKE_DIRECTORY "${LEARN_DEPS_DIR}" )
endif()

# Generate git clone/pull commands (uses runtime checks for rebuild safety)
GitCloneOrPullCmd( PYDENSECRF_CLONE_CMD
  https://github.com/lucasb-eyer/pydensecrf.git ${PYDENSECRF_DIR} )
GitCloneOrPullCmd( PANOPTICAPI_CLONE_CMD
  https://github.com/cocodataset/panopticapi.git ${PANOPTICAPI_DIR} )
GitCloneOrPullCmd( LEARN_CLONE_CMD
  https://gitlab.kitware.com/darpa_learn/learn.git ${LEARN_DIR} viame/master )

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
  # This is only required for no symlink install without a -e with older
  # versions of pip, for never versions the above command works with no -e
  set( LEARN_BUILD_CMD
    ${CMAKE_COMMAND} -E env "${LEARN_ENV_VARS}"
    ${Python_EXECUTABLE} setup.py bdist_wheel -d ${LEARN_BUILD_DIR} )
  set( LEARN_INSTALL_CMD
    ${CMAKE_COMMAND} -E env "${LEARN_ENV_VARS}"
    ${CMAKE_COMMAND} -DWHEEL_DIR=${LEARN_BUILD_DIR}
    -DPYTHON_EXECUTABLE=${Python_EXECUTABLE} -DPython_EXECUTABLE=${Python_EXECUTABLE}
    -P ${VIAME_CMAKE_DIR}/install_python_wheel.cmake )
endif()

set( REMAX_BUILD_CMD
    ${CMAKE_COMMAND} -E env "${LEARN_ENV_VARS}"
    ${Python_EXECUTABLE} setup.py build install )

# Install required dependencies and learn repository
ExternalProject_Add( learn
    DEPENDS python-deps detectron2 torchvideo
    PREFIX ${VIAME_BUILD_PREFIX}
    SOURCE_DIR ${VIAME_PACKAGES_DIR}
    CONFIGURE_COMMAND ${PYDENSECRF_CLONE_CMD}
      COMMAND ${PANOPTICAPI_CLONE_CMD}
      COMMAND ${LEARN_CLONE_CMD}
    BUILD_COMMAND ${LEARN_REQ_PIP_CMD} -r ${LEARN_DIR}/requirements.txt
      COMMAND ${LEARN_REQ_PIP_CMD} -r ${REMAX_DIR}/requirements.txt
      COMMAND cd ${PYDENSECRF_DIR} && ${LEARN_BUILD_CMD}
      COMMAND cd ${PANOPTICAPI_DIR} && ${LEARN_BUILD_CMD}
      COMMAND cd ${REMAX_OPS_DIR} && ${LEARN_BUILD_CMD}
      COMMAND cd ${LEARN_DIR} && ${LEARN_BUILD_CMD}
    INSTALL_COMMAND ${LEARN_INSTALL_CMD}
    LIST_SEPARATOR "----"
    )
