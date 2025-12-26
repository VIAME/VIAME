
set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} learn )

set( LEARN_DIR ${VIAME_SOURCE_DIR}/packages/learn )
set( LEARN_DEPS_DIR ${LEARN_DIR}-deps )

set( PYDENSECRF_DIR ${LEARN_DEPS_DIR}/pydensecrf )
set( PANOPTICAPI_DIR ${LEARN_DEPS_DIR}/panopticapi )
set( REMAX_DIR ${VIAME_SOURCE_DIR}/plugins/pytorch/remax )
set( REMAX_OPS_DIR ${REMAX_DIR}/model/ops )

set( LEARN_BUILD_DIR ${VIAME_BUILD_PREFIX}/src/learn-build )
set( LEARN_CLONE_CMD )

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

if( NOT EXISTS "${PYDENSECRF_DIR}" )
  set( LEARN_CLONE_CMD git clone https://github.com/lucasb-eyer/pydensecrf.git ${PYDENSECRF_DIR} )
else()
  set( LEARN_CLONE_CMD git -C ${PYDENSECRF_DIR} pull )
endif()

if( NOT EXISTS "${PANOPTICAPI_DIR}" )
  set( LEARN_CLONE_CMD ${LEARN_CLONE_CMD} &&
    git clone https://github.com/cocodataset/panopticapi.git ${PANOPTICAPI_DIR} )
else()
  set( LEARN_CLONE_CMD ${LEARN_CLONE_CMD} &&
    git -C ${PANOPTICAPI_DIR} pull )
endif()

if( NOT EXISTS "${LEARN_DIR}" )
  set( LEARN_CLONE_CMD ${LEARN_CLONE_CMD} &&
    git clone --branch viame/master https://gitlab.kitware.com/darpa_learn/learn.git ${LEARN_DIR} )
else()
  set( LEARN_CLONE_CMD ${LEARN_CLONE_CMD} &&
    git -C ${LEARN_DIR} pull )
endif()

# Setup python env vars and commands
set( LEARN_REQ_PIP_CMD
    ${CMAKE_COMMAND} -E env "${LEARN_ENV_VARS}"
    ${Python_EXECUTABLE} -m pip install --user )

# Use modern python -m build instead of deprecated setup.py calls
if( VIAME_PYTHON_SYMLINK )
  # In development mode, pip install -e handles both build and install
  set( LEARN_BUILD_AND_INSTALL_CMD
    ${CMAKE_COMMAND} -E env "${LEARN_ENV_VARS}"
    ${Python_EXECUTABLE} -m pip install --user -e . )
  set( LEARN_INSTALL_CMD "" )
else()
  # Use python -m build for PEP 517 compliant wheel building
  set( LEARN_BUILD_AND_INSTALL_CMD
    ${CMAKE_COMMAND} -E env "${LEARN_ENV_VARS}"
    ${Python_EXECUTABLE} -m build --wheel --no-isolation --outdir ${LEARN_BUILD_DIR} )
  set( LEARN_INSTALL_CMD
    ${CMAKE_COMMAND} -E env "${LEARN_ENV_VARS}"
    ${CMAKE_COMMAND} -DWHEEL_DIR=${LEARN_BUILD_DIR}
    -DPYTHON_EXECUTABLE=${Python_EXECUTABLE} -DPython_EXECUTABLE=${Python_EXECUTABLE}
    -P ${VIAME_CMAKE_DIR}/install_python_wheel.cmake )
endif()

# REMAX uses pip install directly instead of deprecated setup.py install
set( REMAX_BUILD_AND_INSTALL_CMD
    ${CMAKE_COMMAND} -E env "${LEARN_ENV_VARS}"
    ${Python_EXECUTABLE} -m pip install --user . )

if( Python_VERSION VERSION_LESS "3.7" )
  set( FINAL_PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${VIAME_PATCHES_DIR}/timm
        ${VIAME_PYTHON_INSTALL}/site-packages/timm )
else()
  set( FINAL_PATCH_COMMAND )
endif()

# Install required dependencies and learn repository
ExternalProject_Add( learn
    DEPENDS python-deps detectron2 torchvideo
    PREFIX ${VIAME_BUILD_PREFIX}
    SOURCE_DIR ${VIAME_PACKAGES_DIR}
    CONFIGURE_COMMAND "${LEARN_CLONE_CMD}"
    BUILD_COMMAND ${LEARN_REQ_PIP_CMD} -r ${LEARN_DIR}/requirements.txt
          COMMAND ${LEARN_REQ_PIP_CMD} -r ${REMAX_DIR}/requirements.txt
          COMMAND cd ${PYDENSECRF_DIR} && ${LEARN_BUILD_AND_INSTALL_CMD}
          COMMAND cd ${PANOPTICAPI_DIR} && ${LEARN_BUILD_AND_INSTALL_CMD}
          COMMAND cd ${REMAX_OPS_DIR} && ${REMAX_BUILD_AND_INSTALL_CMD}
          COMMAND cd ${LEARN_DIR} && ${LEARN_BUILD_AND_INSTALL_CMD}
          COMMAND ${FINAL_PATCH_COMMAND}
    INSTALL_COMMAND ${LEARN_INSTALL_CMD}
    LIST_SEPARATOR "----"
    )
