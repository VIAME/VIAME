message("Adding LEARN")

# Python stuff
set( PYTHON_DEP_PIP_CMD pip install --user )
string( REPLACE " " ";" PYTHON_DEP_PIP_CMD "${PYTHON_DEP_PIP_CMD}" )

set( PYTHON_LEARN_DEP_BUILD
    ${CMAKE_COMMAND} -E env "${PYTHON_DEP_ENV_VARS}"
    ${Python_EXECUTABLE} -m ${PYTHON_DEP_PIP_CMD} )

# Install learn
set( LEARN_DIR ${VIAME_SOURCE_DIR}/packages/learn )
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
    # git repos
    execute_process(
        COMMAND
            ${PYTHON_LEARN_DEP_BUILD} 
            --upgrade --force-reinstall Pillow
    )
endif()
