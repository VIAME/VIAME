
set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} learn )

set( LEARN_DIR ${VIAME_SOURCE_DIR}/packages/learn )

# Setup python env vars and commands
set( LEARN_DEP_PIP_CMD
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
    DEPENDS python-deps detectron2
    PREFIX ${VIAME_BUILD_PREFIX}
    SOURCE_DIR ${LEARN_DIR}
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${LEARN_DEP_PIP_CMD} -r ${LEARN_DIR}/requirements.txt
          COMMAND ${LEARN_DEP_PIP_CMD}
                  "git+https://github.com/lucasb-eyer/pydensecrf.git" 
                  "git+https://github.com/cocodataset/panopticapi.git"
    INSTALL_COMMAND ${LEARN_PIP_CMD}
    LIST_SEPARATOR "----"
    )
