# PyTorch External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} pytorch )

CreateDirectory( ${VIAME_BUILD_PREFIX}/src/pytorch-build )
CreateDirectory( ${VIAME_BUILD_PREFIX}/src/pytorch-build/pip-tmp )

set( PYTORCH_PIP_BUILD_DIR_CMD -b ${VIAME_BUILD_PREFIX}/src/pytorch-build/pip-build )
set( PYTORCH_PIP_CACHE_DIR_CMD --cache-dir ${VIAME_BUILD_PREFIX}/src/pytorch-build/pip-cache )
set( PYTORCH_PIP_TMP_DIR ${VIAME_BUILD_PREFIX}/src/pytorch-build/pip-tmp )

set( PYTORCH_PIP_SETTINGS ${PYTORCH_PIP_BUILD_DIR_CMD} ${PYTORCH_PIP_CACHE_DIR_CMD} )

if( VIAME_SYMLINK_PYTHON )
  set( PYTORCH_PIP_CMD
    pip install ${PYTORCH_PIP_SETTINGS} --user -e . )
  set( TORCHVISION_PIP_CMD
    pip install ${PYTORCH_PIP_SETTINGS} --user -e . )
else()
  set( PYTORCH_PIP_CMD
    pip install ${PYTORCH_PIP_SETTINGS} --user file://${VIAME_PACKAGES_DIR}/pytorch )
  set( TORCHVISION_PIP_CMD
    pip install ${PYTORCH_PIP_SETTINGS} --user file://${VIAME_PACKAGES_DIR}/torchvision )
endif()

if( VIAME_ENABLE_CUDNN )
  set(CUDNN_ENV "CUDNN_LIBRARY=${CUDNN_LIBRARIES}")
  if(C UDNN_ROOT_DIR )
    list(APPEND CUDNN_ENV "CUDNN_INCLUDE_DIR=${CUDNN_ROOT_DIR}/include")
  endif()
else()
  unset(CUDNN_ENV)
endif()

if( VIAME_ENABLE_CUDA )
  set( PYTORCH_CUDA_ARCHITECTURES "3.0 3.5 5.0 5.2 6.0 6.1+PTX" )
  set( TORCH_NVCC_FLAGS "-Xfatbin -compress-all" )
else()
  set( PYTORCH_CUDA_ARCHITECTURES )
  set( TORCH_NVCC_FLAGS )
endif()

set( PYTHON_BASEPATH
  ${VIAME_BUILD_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}${PYTHON_ABIFLAGS} )
set( CUSTOM_PYTHONPATH
  ${PYTHON_BASEPATH}/site-packages:${PYTHON_BASEPATH}/dist-packages )
set( CUSTOM_PATH
  ${VIAME_BUILD_INSTALL_PREFIX}/bin:$ENV{PATH} )
set( PYTORCH_PYTHON_INSTALL
  ${CMAKE_COMMAND} -E env PYTHONPATH=${CUSTOM_PYTHONPATH}
                      TMPDIR=${PYTORCH_PIP_TMP_DIR}
                      PATH=${CUSTOM_PATH}
                      PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}
                      CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}
                      TORCH_CUDA_ARCH_LIST=${PYTORCH_CUDA_ARCHITECTURES}
                      TORCH_NVCC_FLAGS=${PYTORCH_NVCC_FLAGS}
                      ${CUDNN_ENV}
    ${PYTHON_EXECUTABLE} -m ${PYTORCH_PIP_CMD} )
set( TORCHVISION_PYTHON_INSTALL
  ${CMAKE_COMMAND} -E env PYTHONPATH=${CUSTOM_PYTHONPATH}
                      TMPDIR=${PYTORCH_PIP_TMP_DIR}
                      PATH=${CUSTOM_PATH}
                      PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}
                      CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}
                      ${CUDNN_ENV}
    ${PYTHON_EXECUTABLE} -m ${TORCHVISION_PIP_CMD} )

ExternalProject_Add( pytorch
  DEPENDS fletch
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/pytorch
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ${PYTORCH_PYTHON_INSTALL}
  )

ExternalProject_Add_Step(pytorch install_torchvision
  WORKING_DIRECTORY ${VIAME_PACKAGES_DIR}/torchvision
  COMMAND ${TORCHVISION_PYTHON_INSTALL}
  COMMENT "Installing torchvision python files."
  DEPENDEES install
  )

if ( VIAME_FORCEBUILD )
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
