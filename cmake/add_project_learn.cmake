
set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} learn )

set( LEARN_DIR ${VIAME_SOURCE_DIR}/packages/learn )

set( PYDENSECRF_DIR ${LEARN_DIR}-deps/pydensecrf )
set( PANOPTICAPI_DIR ${LEARN_DIR}-deps/panopticapi )

set( LEARN_CLONE_CMD
  git clone https://github.com/lucasb-eyer/pydensecrf.git ${PYDENSECRF_DIR} &&
  git clone https://github.com/cocodataset/panopticapi.git ${PANOPTICAPI_DIR} &&
  git clone --branch viame/master https://gitlab.kitware.com/darpa_learn/learn.git ${LEARN_DIR} )

# Setup python env vars and commands
set( LEARN_REQ_PIP_CMD
    ${CMAKE_COMMAND} -E env "${PYTHON_DEP_ENV_VARS}"
    ${Python_EXECUTABLE} -m pip install --user )

if( VIAME_SYMLINK_PYTHON )
  set( LEARN_PIP_CMD
    ${CMAKE_COMMAND} -E env "${PYTHON_DEP_ENV_VARS}"
    ${Python_EXECUTABLE} -m pip install --user -e . )
else()
  # This is only required for no symlink install without a -e with older
  # versions of pip, for never versions the above command works with no -e
  set( LEARN_PIP_CMD
      ${CMAKE_COMMAND} -E env "${PYTHON_DEP_ENV_VARS}"
      ${Python_EXECUTABLE} -m pip install --user ${LEARN_DIR} )
endif()

# Install required dependencies and learn
ExternalProject_Add( learn
    DEPENDS python-deps detectron2 torchvideo
    PREFIX ${VIAME_BUILD_PREFIX}
    SOURCE_DIR ${LEARN_DIR}
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ${LEARN_CLONE_CMD}
    BUILD_COMMAND ${LEARN_REQ_PIP_CMD} -r ${LEARN_DIR}/requirements.txt
          COMMAND ${LEARN_BUILD_CMD} ${LEARN_DIR}
          COMMAND ${LEARN_BUILD_CMD} ${PYDENSECRF_DIR}
          COMMAND ${LEARN_BUILD_CMD} ${PANOPTICAPI_DIR}
    INSTALL_COMMAND ${LEARN_INSTALL_CMD}
    LIST_SEPARATOR "----"
    )
