# PyTorch External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} pytorch )

if( VIAME_SYMLINK_PYTHON )
  set( PYTORCH_PIP_CMD
    pip install --user -e . )
  set( TORCHVISION_PIP_CMD
    pip install --user -e . )
else()
  set( PYTORCH_PIP_CMD
    pip install --user file://${VIAME_PACKAGES_DIR}/pytorch )
  set( TORCHVISION_PIP_CMD
    pip install --user file://${VIAME_PACKAGES_DIR}/torchvision )
endif()

set( PYTHON_BASEPATH
  ${VIAME_BUILD_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}${PYTHON_ABIFLAGS} )
set( CUSTOM_PYTHONPATH
  ${PYTHON_BASEPATH}/site-packages:${PYTHON_BASEPATH}/dist-packages )
set( CUSTOM_PATH
  ${VIAME_BUILD_INSTALL_PREFIX}/bin:$ENV{PATH} )
set( PYTORCH_PYTHON_BUILD
  ${CMAKE_COMMAND} -E env PYTHONPATH=${CUSTOM_PYTHONPATH}
                      PATH=${CUSTOM_PATH}
                      PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}
                      CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}
    ${PYTHON_EXECUTABLE} -m ${PYTORCH_PIP_CMD} )
set( TORCHVISION_PYTHON_INSTALL
  ${CMAKE_COMMAND} -E env PYTHONPATH=${CUSTOM_PYTHONPATH}
                      PATH=${CUSTOM_PATH}
                      PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}
                      CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}
    ${PYTHON_EXECUTABLE} -m ${TORCHVISION_PIP_CMD} )

ExternalProject_Add( pytorch
  DEPENDS fletch
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/pytorch
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${PYTORCH_PYTHON_BUILD}
  INSTALL_COMMAND ""
  )

ExternalProject_Add_Step(pytorch install_torchvision
  WORKING_DIRECTORY ${VIAME_PACKAGES_DIR}/torchvision
  COMMAND ${TORCHVISION_PYTHON_INSTALL}
  COMMENT "Installing torchvision python files."
  DEPENDEES build
  )

if (VIAME_FORCEBUILD)
ExternalProject_Add_Step(pytorch forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${VIAME_BUILD_PREFIX}/src/pytorch-stamp/pytorch-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )
endif()

set(VIAME_ARGS_pytorch
  -Dpytorch_DIR:PATH=${VIAME_BUILD_PREFIX}/src/pytorch-build
  )
