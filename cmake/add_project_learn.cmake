
set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} learn )

# Setup python env vars and commands
set( PYTHON_DEP_PIP_CMD pip install --user )
string( REPLACE " " ";" PYTHON_DEP_PIP_CMD "${PYTHON_DEP_PIP_CMD}" )

set( PYTHON_LEARN_DEP_BUILD
    ${CMAKE_COMMAND} -E env "${PYTHON_DEP_ENV_VARS}"
    ${Python_EXECUTABLE} -m ${PYTHON_DEP_PIP_CMD} )

# Install required dependencies and learn
set( LEARN_DIR ${VIAME_SOURCE_DIR}/packages/learn )

ExternalProject_Add( learn
    DEPENDS python-deps detectron2
    PREFIX ${VIAME_BUILD_PREFIX}
    SOURCE_DIR ${LEARN_DIR}
    BUILD_IN_SOURCE 1
    PATCH_COMMAND ${LIBRARY_PATCH_COMMAND}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${PYTHON_LEARN_DEP_BUILD} -r ${LEARN_DIR}/requirements.txt
          COMMAND ${PYTHON_LEARN_DEP_BUILD}
                  "git+https://github.com/lucasb-eyer/pydensecrf.git" 
                  "git+https://github.com/cocodataset/panopticapi.git"
                  "git+https://github.com/mcordts/cityscapesScripts.git"
    INSTALL_COMMAND ${PYTHON_LEARN_DEP_BUILD} -e ${LEARN_DIR}
    LIST_SEPARATOR "----"
    )
