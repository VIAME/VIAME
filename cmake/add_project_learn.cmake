message("Adding LEARN")

# Python stuff
set( PYTHON_DEP_PIP_CMD pip install --user )
string( REPLACE " " ";" PYTHON_DEP_PIP_CMD "${PYTHON_DEP_PIP_CMD}" )

set( PYTHON_LEARN_DEP_BUILD
    ${CMAKE_COMMAND} -E env "${PYTHON_DEP_ENV_VARS}"
    ${Python_EXECUTABLE} -m ${PYTHON_DEP_PIP_CMD} )

# Install learn
set( LEARN_DIR ${VIAME_SOURCE_DIR}/packages/learn )
ExternalProject_Add( learn_requirements
    DEPENDS ${PROJECT_DEPS}
    PREFIX ${VIAME_BUILD_PREFIX}
    SOURCE_DIR ${LEARN_DIR}
    BUILD_IN_SOURCE 1
    PATCH_COMMAND ${LIBRARY_PATCH_COMMAND}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${PYTHON_LEARN_DEP_BUILD} -r ${LEARN_DIR}/requirements.txt
    INSTALL_COMMAND ${LIBRARY_PYTHON_INSTALL}
    LIST_SEPARATOR "----"
    )

ExternalProject_Add( learn
    DEPENDS ${PROJECT_DEPS}
    PREFIX ${VIAME_BUILD_PREFIX}
    SOURCE_DIR ${LEARN_DIR}
    BUILD_IN_SOURCE 1
    PATCH_COMMAND ${LIBRARY_PATCH_COMMAND}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${PYTHON_LEARN_DEP_BUILD} -e ${LEARN_DIR}
    INSTALL_COMMAND ${LIBRARY_PYTHON_INSTALL}
    LIST_SEPARATOR "----"
    )
execute_process(
    COMMAND
        ${PYTHON_LEARN_DEP_BUILD} 
        -r ${LEARN_DIR}/requirements.txt
)
execute_process(
    COMMAND
        ${PYTHON_LEARN_DEP_BUILD} 
        -e ${LEARN_DIR}
)

#####################
# Individual algos
#####################
# CutLer
if( VIAME_ENABLE_PYTORCH-CUTLER )
    set(CUTLER_DIR ${LEARN_DIR}/learn/algorithms/CutLER/CutLER_main)

    # git repos
    execute_process(
        COMMAND
            ${PYTHON_LEARN_DEP_BUILD} 
            "git+https://github.com/lucasb-eyer/pydensecrf.git" 
            "git+https://github.com/cocodataset/panopticapi.git"
            "git+https://github.com/mcordts/cityscapesScripts.git"
    )
    execute_process(
        COMMAND
            ${PYTHON_LEARN_DEP_BUILD} 
            --upgrade --force-reinstall Pillow
    )
    # mmdet for training
    execute_process(
        COMMAND
            MMCV_WITH_OPS=1 FORCE_CUDA=1 
            ${PYTHON_LEARN_DEP_BUILD} 
            mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
            
    )
    execute_process(
        COMMAND
            ${PYTHON_LEARN_DEP_BUILD} 
            mmdet==2.25.0
    )
endif()
