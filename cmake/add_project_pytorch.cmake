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
CreateDirectory( ${VIAME_BUILD_PREFIX}/src/torchvision-build )
CreateDirectory( ${VIAME_BUILD_PREFIX}/src/torchvision-build/pip-tmp )

set( PYTHON_PIP_COMMAND ${PYTHON_EXECUTABLE} -m pip )

set( PYTORCH_PIP_BUILD_DIR_CMD -b ${VIAME_BUILD_PREFIX}/src/pytorch-build/pip-build )
set( PYTORCH_PIP_CACHE_DIR_CMD --cache-dir ${VIAME_BUILD_PREFIX}/src/pytorch-build/pip-cache )
set( PYTORCH_PIP_TMP_DIR ${VIAME_BUILD_PREFIX}/src/pytorch-build/pip-tmp )
set( PYTORCH_PIP_TMP_DIR ${VIAME_BUILD_PREFIX}/src/torchvision-build/pip-tmp )

set( PYTORCH_PIP_SETTINGS ${PYTORCH_PIP_BUILD_DIR_CMD} ${PYTORCH_PIP_CACHE_DIR_CMD} )

if( VIAME_SYMLINK_PYTHON )
  set( PYTORCH_PIP_CMD
    pip install ${PYTORCH_PIP_SETTINGS} --user -e . )
  set( TORCHVISION_PIP_CMD
    pip install ${PYTORCH_PIP_SETTINGS} --user -e . )

  set( PYTORCH_PIP_BUILD_CMD
    ${PYTHON_EXECUTABLE} setup.py build )
  set( PYTORCH_PIP_INSTALL_CMD
    ${PYTHON_PIP_COMMAND} install --user -e . )

  set( TORCHVISION_PIP_BUILD_CMD
    ${PYTHON_EXECUTABLE} setup.py build )
  set( TORCHVISION_PIP_INSTALL_CMD
    ${PYTHON_PIP_COMMAND} install --user -e . )
else()
  set( PYTORCH_PIP_CMD
    pip install ${PYTORCH_PIP_SETTINGS} --user file://${VIAME_PACKAGES_DIR}/pytorch )
  set( TORCHVISION_PIP_CMD
    pip install ${PYTORCH_PIP_SETTINGS} --user file://${VIAME_PACKAGES_DIR}/pytorch-libs/torchvision )

  set( PYTORCH_PIP_BUILD_CMD
    ${PYTHON_EXECUTABLE} setup.py bdist_wheel -d ${VIAME_BUILD_PREFIX}/src/pytorch-build )
  set( PYTORCH_PIP_INSTALL_CMD
    ${CMAKE_COMMAND}
      -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
      -DWHEEL_DIR=${VIAME_BUILD_PREFIX}/src/pytorch-build
      -P ${CMAKE_SOURCE_DIR}/cmake/install_python_wheel.cmake )

  set( TORCHVISION_PIP_BUILD_CMD
    ${PYTHON_EXECUTABLE} setup.py bdist_wheel -d ${VIAME_BUILD_PREFIX}/src/torchvision-build )
  set( TORCHVISION_PIP_INSTALL_CMD
    ${CMAKE_COMMAND}
      -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
      -DWHEEL_DIR=${VIAME_BUILD_PREFIX}/src/torchvision-build
      -P ${CMAKE_SOURCE_DIR}/cmake/install_python_wheel.cmake )
endif()

if( VIAME_ENABLE_CUDNN )
  set(CUDNN_ENV "CUDNN_LIBRARY=${CUDNN_LIBRARIES}")
  if( CUDNN_ROOT_DIR )
    list(APPEND CUDNN_ENV "CUDNN_INCLUDE_DIR=${CUDNN_ROOT_DIR}/include")
  endif()
else()
  unset(CUDNN_ENV)
endif()

if( VIAME_ENABLE_CUDA )
  set( PYTORCH_CUDA_ARCHITECTURES "3.0 3.5 5.0 5.2 6.0 6.1+PTX" )
  set( TORCH_NVCC_FLAGS "-Xfatbin -compress-all" )
  set( CUSTOM_PATH ${VIAME_BUILD_INSTALL_PREFIX}/bin:${CUDA_TOOLKIT_ROOT_DIR}/bin:$ENV{PATH} )
else()
  set( PYTORCH_CUDA_ARCHITECTURES )
  set( TORCH_NVCC_FLAGS )
  set( CUSTOM_PATH ${VIAME_BUILD_INSTALL_PREFIX}/bin:$ENV{PATH} )
endif()

set( PYTHON_BASEPATH
  ${VIAME_BUILD_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}${PYTHON_ABIFLAGS} )
set( CUSTOM_PYTHONPATH
  ${PYTHON_BASEPATH}/site-packages:${PYTHON_BASEPATH}/dist-packages )
set( PYTORCH_PYTHON_BUILD
  ${CMAKE_COMMAND} -E env PYTHONPATH=${CUSTOM_PYTHONPATH}
                      TMPDIR=${PYTORCH_PIP_TMP_DIR}
                      PATH=${CUSTOM_PATH}
                      PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}
                      CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}
                      TORCH_CUDA_ARCH_LIST=${PYTORCH_CUDA_ARCHITECTURES}
                      TORCH_NVCC_FLAGS=${PYTORCH_NVCC_FLAGS}
                      ${CUDNN_ENV}
    ${PYTORCH_PIP_BUILD_CMD} )
set( PYTORCH_PYTHON_INSTALL
  ${CMAKE_COMMAND} -E env PYTHONPATH=${CUSTOM_PYTHONPATH}
                      TMPDIR=${PYTORCH_PIP_TMP_DIR}
                      PATH=${CUSTOM_PATH}
                      PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}
                      CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}
                      TORCH_CUDA_ARCH_LIST=${PYTORCH_CUDA_ARCHITECTURES}
                      TORCH_NVCC_FLAGS=${PYTORCH_NVCC_FLAGS}
                      ${CUDNN_ENV}
    ${PYTORCH_PIP_INSTALL_CMD} )
set( TORCHVISION_PYTHON_BUILD
  ${CMAKE_COMMAND} -E env PYTHONPATH=${CUSTOM_PYTHONPATH}
                      TMPDIR=${PYTORCH_PIP_TMP_DIR}
                      PATH=${CUSTOM_PATH}
                      PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}
                      CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}
                      ${CUDNN_ENV}
    ${TORCHVISION_PIP_BUILD_CMD} )
set( TORCHVISION_PYTHON_INSTALL
  ${CMAKE_COMMAND} -E env PYTHONPATH=${CUSTOM_PYTHONPATH}
                      TMPDIR=${PYTORCH_PIP_TMP_DIR}
                      PATH=${CUSTOM_PATH}
                      PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}
                      CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}
                      ${CUDNN_ENV}
    ${TORCHVISION_PIP_INSTALL_CMD} )

ExternalProject_Add( pytorch
  DEPENDS fletch
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/pytorch
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${PYTORCH_PYTHON_BUILD}
  INSTALL_COMMAND ${PYTORCH_PYTHON_INSTALL}
  )

ExternalProject_Add( torchvision
  DEPENDS fletch pytorch
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/pytorch-libs/torchvision
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${TORCHVISION_PYTHON_BUILD}
  INSTALL_COMMAND ${TORCHVISION_PYTHON_INSTALL}
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
  ExternalProject_Add_Step(torchvision forcebuild
    COMMAND ${CMAKE_COMMAND}
      -E remove ${VIAME_BUILD_PREFIX}/src/torchvision-stamp/torchvision-build
    COMMENT "Removing build stamp file for build update (forcebuild)."
    DEPENDEES configure
    DEPENDERS build
    ALWAYS 1
    )
endif()

set(VIAME_ARGS_pytorch
  -Dpytorch_DIR:PATH=${VIAME_BUILD_PREFIX}/src/pytorch-build
  )
