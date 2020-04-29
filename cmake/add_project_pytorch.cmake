# PyTorch External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

CreateDirectory( ${VIAME_BUILD_PREFIX}/src/pytorch-build )

set( PYTORCH_LIBS_TO_BUILD )

if( VIAME_ENABLE_PYTORCH-INTERNAL )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} pytorch )

  set( COMMON_PYTORCH_PROJECT_DEP fletch pytorch )
else()
  set( COMMON_PYTORCH_PROJECT_DEP fletch torch )
endif()

if( VIAME_ENABLE_PYTORCH AND ( NOT WIN32 OR VIAME_ENABLE_CUDA ) )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} torchvision )
endif()

if( VIAME_ENABLE_PYTORCH-MMDET )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} mmcv mmdetection )
endif()

if( VIAME_ENABLE_PYTORCH-PYSOT )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} pysot )
endif()

if( VIAME_ENABLE_PYTORCH-MDNET )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} roi-align )
endif()

if( VIAME_ENABLE_PYTORCH-NETHARN )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} netharn bioharn )
endif()

if( VIAME_ENABLE_PYTORCH-MMDET )
  if( VIAME_ENABLE_CUDA AND CUDA_VERSION VERSION_LESS "9.0" )
    message( FATAL_ERROR "To use mmdetection you must have at least CUDA 9.0.\n\n"
                         "Install CUDA 9.0+ or disable VIAME_ENABLE_PYTORCH-MMDET" )
  endif()
  if( NOT VIAME_ENABLE_PYTHON-INTERNAL AND PYTHON_VERSION VERSION_LESS "3.0" )
    message( FATAL_ERROR "To use mmdetection you must have at least Python 3.0.\n\n"
                         "Use Python3 or disable VIAME_ENABLE_PYTORCH-MMDET" )
  endif()
endif()

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} ${PYTORCH_LIBS_TO_BUILD} )

if( VIAME_ENABLE_CUDNN )
  if( VIAME_ENABLE_PYTORCH-INTERNAL AND "${CUDNN_VERSION_MAJOR}" VERSION_LESS "7.0.0" )
    message( FATAL_ERROR "CUDNN version 7.0 or higher required for internal pytorch" )
  endif()

  set( EXTRA_ENV "CUDNN_LIBRARY=${CUDNN_LIBRARIES}" )
  if( CUDNN_ROOT_DIR )
    list( APPEND EXTRA_ENV "CUDNN_INCLUDE_DIR=${CUDNN_ROOT_DIR}/include" )
  endif()
  if( WIN32 )
    string( REPLACE ";" "----" EXTRA_ENV "${EXTRA_ENV}" )
  endif()
else()
  unset( EXTRA_ENV )
endif()

if( VIAME_ENABLE_PYTORCH-DISABLE-NINJA )
  list( APPEND EXTRA_ENV "USE_NINJA=OFF" )
endif()

if( VIAME_ENABLE_PYTORCH-FORCE-CUDA )
  list( APPEND EXTRA_ENV "FORCE_CUDA=1" )
endif()

set( PYTHON_BASEPATH
  ${VIAME_BUILD_INSTALL_PREFIX}/lib/python${PYTHON_VERSION} )

if( VIAME_ENABLE_CUDA )
  set( TORCH_CUDA_ARCHITECTURES "3.5 5.0 5.2 6.0 6.1" )
  set( TORCH_NVCC_FLAGS "-Xfatbin -compress-all" )

  if( CUDA_VERSION VERSION_GREATER "8.5" )
    set( TORCH_CUDA_ARCHITECTURES "${TORCH_CUDA_ARCHITECTURES} 7.0 7.0+PTX" )
  endif()
  if( CUDA_VERSION VERSION_GREATER "9.5" )
    set( TORCH_CUDA_ARCHITECTURES "${TORCH_CUDA_ARCHITECTURES} 7.5 7.5+PTX" )
  endif()
  if( CUDA_VERSION VERSION_LESS "9.0" AND VIAME_ENABLE_PYTORCH-NETHARN )
    message( FATAL_ERROR "VIAME_ENABLE_PYTORCH-NETHARN requires CUDA 9 or above" )
  endif()
else()
  set( TORCH_CUDA_ARCHITECTURES )
  set( TORCH_NVCC_FLAGS )
endif()

if( WIN32 )
  set( CUSTOM_PYTHONPATH
    ${PYTHON_BASEPATH};${PYTHON_BASEPATH}/site-packages;${PYTHON_BASEPATH}/dist-packages )
  if( VIAME_ENABLE_CUDA )
    set( CUSTOM_PATH
      ${VIAME_BUILD_INSTALL_PREFIX}/bin;${CUDA_TOOLKIT_ROOT_DIR}/bin;$ENV{PATH} )
  else()
    set( CUSTOM_PATH
      ${VIAME_BUILD_INSTALL_PREFIX}/bin )
  endif()
  string( REPLACE ";" "----" CUSTOM_PYTHONPATH "${CUSTOM_PYTHONPATH}" )
  string( REPLACE ";" "----" CUSTOM_PATH "${CUSTOM_PATH}" )
else()
  set( CUSTOM_PYTHONPATH
    ${PYTHON_BASEPATH}:${PYTHON_BASEPATH}/site-packages:${PYTHON_BASEPATH}/dist-packages )
  if( VIAME_ENABLE_CUDA )
    set( CUSTOM_PATH
      ${VIAME_BUILD_INSTALL_PREFIX}/bin:${CUDA_TOOLKIT_ROOT_DIR}/bin:$ENV{PATH} )
  else()
    set( CUSTOM_PATH
      ${VIAME_BUILD_INSTALL_PREFIX}/bin:$ENV{PATH} )
  endif()
endif()

foreach( LIB ${PYTORCH_LIBS_TO_BUILD} )

  if( "${LIB}" STREQUAL "pytorch" )
    set( LIBRARY_LOCATION ${VIAME_PACKAGES_DIR}/pytorch )
    set( LIBRARY_LOCATION_URL file://${LIBRARY_LOCATION} )
  elseif( "${LIB}" STREQUAL "roi-align" )
    set( LIBRARY_LOCATION ${VIAME_SOURCE_DIR}/plugins/pytorch/mdnet )
    set( LIBRARY_LOCATION_URL file://${LIBRARY_LOCATION} )
  else()
    set( LIBRARY_LOCATION ${VIAME_PACKAGES_DIR}/pytorch-libs/${LIB} )
    set( LIBRARY_LOCATION_URL file://${LIBRARY_LOCATION} )
  endif()

  set( LIBRARY_PIP_CACHE_DIR ${VIAME_BUILD_PREFIX}/src/pytorch-build/${LIB}-cache )
  set( LIBRARY_PIP_BUILD_DIR ${VIAME_BUILD_PREFIX}/src/pytorch-build/${LIB}-build )
  set( LIBRARY_PIP_TMP_DIR ${VIAME_BUILD_PREFIX}/src/pytorch-build/${LIB}-tmp )

  CreateDirectory( ${LIBRARY_PIP_CACHE_DIR} )
  CreateDirectory( ${LIBRARY_PIP_BUILD_DIR} )
  CreateDirectory( ${LIBRARY_PIP_TMP_DIR} )

  set( LIBRARY_PIP_BUILD_DIR_CMD -b ${LIBRARY_PIP_BUILD_DIR} )
  set( LIBRARY_PIP_CACHE_DIR_CMD --cache-dir ${LIBRARY_PIP_CACHE_DIR} )

  set( LIBRARY_PIP_SETTINGS ${LIBRARY_PIP_BUILD_DIR_CMD} ${LIBRARY_PIP_CACHE_DIR_CMD} )

  if( VIAME_SYMLINK_PYTHON )
    set( LIBRARY_PIP_BUILD_CMD
      ${PYTHON_EXECUTABLE} setup.py build )
    set( LIBRARY_PIP_INSTALL_CMD
      ${PYTHON_EXECUTABLE} -m pip install --user -e . )
  else()
    if( "${LIB}" STREQUAL "mmcv" )
      set( LIBRARY_PIP_BUILD_CMD
        ${PYTHON_EXECUTABLE} setup.py
          bdist_wheel -d ${LIBRARY_PIP_BUILD_DIR} )
    else()
      set( LIBRARY_PIP_BUILD_CMD
        ${PYTHON_EXECUTABLE} setup.py build_ext
          --include-dirs="${VIAME_BUILD_INSTALL_PREFIX}/include"
          --library-dirs="${VIAME_BUILD_INSTALL_PREFIX}/lib"
          --inplace bdist_wheel -d ${LIBRARY_PIP_BUILD_DIR} )
    endif()
    set( LIBRARY_PIP_INSTALL_CMD
      ${CMAKE_COMMAND}
        -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
        -DWHEEL_DIR=${LIBRARY_PIP_BUILD_DIR}
        -P ${VIAME_CMAKE_DIR}/install_python_wheel.cmake )
  endif()

  set( LIBRARY_PYTHON_BUILD
    ${CMAKE_COMMAND} -E env "PYTHONPATH=${CUSTOM_PYTHONPATH}"
                            "TMPDIR=${LIBRARY_PIP_TMP_DIR}"
                            "PATH=${CUSTOM_PATH}"
                            "PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}"
                            "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}"
                            "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCHITECTURES}"
                            "TORCH_NVCC_FLAGS=${TORCH_NVCC_FLAGS}"
                            "${EXTRA_ENV}"
      ${LIBRARY_PIP_BUILD_CMD} )
  set( LIBRARY_PYTHON_INSTALL
    ${CMAKE_COMMAND} -E env "PYTHONPATH=${CUSTOM_PYTHONPATH}"
                            "TMPDIR=${LIBRARY_PIP_TMP_DIR}"
                            "PATH=${CUSTOM_PATH}"
                            "PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}"
                            "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}"
                            "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCHITECTURES}"
                            "TORCH_NVCC_FLAGS=${TORCH_NVCC_FLAGS}"
                            "${EXTRA_ENV}"
      ${LIBRARY_PIP_INSTALL_CMD} )

  if( "${LIB}" STREQUAL "bioharn" )
    set( PROJECT_DEPS ${COMMON_PYTORCH_PROJECT_DEP} netharn )
  elseif( "${LIB}" STREQUAL "netharn" )
    set( PROJECT_DEPS ${COMMON_PYTORCH_PROJECT_DEP} mmdetection )
  elseif( "${LIB}" STREQUAL "mmdetection" )
    set( PROJECT_DEPS ${COMMON_PYTORCH_PROJECT_DEP} mmcv torchvision )
  elseif( "${LIB}" STREQUAL "pytorch" )
    set( PROJECT_DEPS fletch pyyaml pillow )
  else()
    set( PROJECT_DEPS ${COMMON_PYTORCH_PROJECT_DEP} )
  endif()

  ExternalProject_Add( ${LIB}
    DEPENDS ${PROJECT_DEPS}
    PREFIX ${VIAME_BUILD_PREFIX}
    SOURCE_DIR ${LIBRARY_LOCATION}
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${LIBRARY_PYTHON_BUILD}
    INSTALL_COMMAND ${LIBRARY_PYTHON_INSTALL}
    LIST_SEPARATOR "----"
    )

  if( VIAME_FORCEBUILD )
    ExternalProject_Add_Step( ${LIB} forcebuild
      COMMAND ${CMAKE_COMMAND}
        -E remove ${VIAME_BUILD_PREFIX}/src/{LIB}-stamp
      COMMENT "Removing build stamp file for build update (forcebuild)."
      DEPENDEES configure
      DEPENDERS build
      ALWAYS 1
      )
  endif()
endforeach()

set( VIAME_ARGS_pytorch
  -Dpytorch_DIR:PATH=${VIAME_BUILD_PREFIX}/src/pytorch-build
  )
