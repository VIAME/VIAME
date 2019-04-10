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

set( PYTORCH_LIBRARIES pytorch torchvision mmcv mmdetection )

if( VIAME_ENABLE_CUDNN )
  set(CUDNN_ENV "CUDNN_LIBRARY=${CUDNN_LIBRARIES}")
  if( CUDNN_ROOT_DIR )
    list(APPEND CUDNN_ENV "CUDNN_INCLUDE_DIR=${CUDNN_ROOT_DIR}/include")
  endif()
else()
  unset(CUDNN_ENV)
endif()

if( VIAME_ENABLE_CUDA )
  set( TORCH_CUDA_ARCHITECTURES "3.0 3.5 5.0 5.2 6.0 6.1+PTX" )
  set( TORCH_NVCC_FLAGS "-Xfatbin -compress-all" )
  set( CUSTOM_PATH ${VIAME_BUILD_INSTALL_PREFIX}/bin:${CUDA_TOOLKIT_ROOT_DIR}/bin:$ENV{PATH} )
else()
  set( TORCH_CUDA_ARCHITECTURES )
  set( TORCH_NVCC_FLAGS )
  set( CUSTOM_PATH ${VIAME_BUILD_INSTALL_PREFIX}/bin:$ENV{PATH} )
endif()

foreach( LIB ${PYTORCH_LIBRARIES} )

  if( "${LIB}" STREQUAL "pytorch" )
    set( LIBRARY_LOCATION ${VIAME_PACKAGES_DIR}/pytorch )
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
    set( LIBRARY_PIP_BUILD_CMD
      ${PYTHON_EXECUTABLE} setup.py bdist_wheel -d ${LIBRARY_PIP_BUILD_DIR} )
    set( LIBRARY_PIP_INSTALL_CMD
      ${CMAKE_COMMAND}
        -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
        -DWHEEL_DIR=${LIBRARY_PIP_BUILD_DIR}
        -P ${CMAKE_SOURCE_DIR}/cmake/install_python_wheel.cmake )
  endif()

  set( PYTHON_BASEPATH
    ${VIAME_BUILD_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}${PYTHON_ABIFLAGS} )
  set( CUSTOM_PYTHONPATH
    ${PYTHON_BASEPATH}/site-packages:${PYTHON_BASEPATH}/dist-packages )
  set( LIBRARY_PYTHON_BUILD
    ${CMAKE_COMMAND} -E env PYTHONPATH=${CUSTOM_PYTHONPATH}
                        TMPDIR=${LIBRARY_PIP_TMP_DIR}
                        PATH=${CUSTOM_PATH}
                        PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}
                        CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}
                        TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCHITECTURES}
                        TORCH_NVCC_FLAGS=${TORCH_NVCC_FLAGS}
                        ${CUDNN_ENV}
      ${LIBRARY_PIP_BUILD_CMD} )
  set( LIBRARY_PYTHON_INSTALL
    ${CMAKE_COMMAND} -E env PYTHONPATH=${CUSTOM_PYTHONPATH}
                        TMPDIR=${LIBRARY_PIP_TMP_DIR}
                        PATH=${CUSTOM_PATH}
                        PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}
                        CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}
                        TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCHITECTURES}
                        TORCH_NVCC_FLAGS=${TORCH_NVCC_FLAGS}
                        ${CUDNN_ENV}
      ${LIBRARY_PIP_INSTALL_CMD} )

  if( "${LIB}" STREQUAL "mmdetection" )
    set( MMDET_SUBDEPS roi_align roi_pool nms dcn )

    foreach( DEP ${MMDET_SUBDEPS} )
      set( DEP_PYTHON_BUILD
        ${CMAKE_COMMAND} -E env PYTHONPATH=${CUSTOM_PYTHONPATH}
                            TMPDIR=${LIBRARY_PIP_TMP_DIR}
                            PATH=${CUSTOM_PATH}
                            PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}
                            CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}
                            TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCHITECTURES}
                            TORCH_NVCC_FLAGS=${TORCH_NVCC_FLAGS}
                            ${CUDNN_ENV}
          ${PYTHON_EXECUTABLE} setup.py build_ext --inplace )
      ExternalProject_Add( mmdet_${DEP}
        DEPENDS fletch pytorch mmcv
        PREFIX ${VIAME_BUILD_PREFIX}
        SOURCE_DIR ${VIAME_PACKAGES_DIR}/pytorch-libs/${LIB}/mmdet/ops/${DEP}
        STAMP_DIR ${VIAME_BUILD_PREFIX}/src/pytorch-build/${DEP}-stamp
        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ${DEP_PYTHON_BUILD}
        INSTALL_COMMAND ""
        )
    endforeach()

    set( PROJECT_DEPS fletch pytorch mmcv
         mmdet_roi_align mmdet_roi_pool mmdet_dcn mmdet_nms )
  elseif( "${LIB}" STREQUAL "pytorch" )
    set( PROJECT_DEPS fletch )
  else()
    set( PROJECT_DEPS fletch pytorch )
  endif()

  ExternalProject_Add( ${LIB}
    DEPENDS ${PROJECT_DEPS}
    PREFIX ${VIAME_BUILD_PREFIX}
    SOURCE_DIR ${LIBRARY_LOCATION}
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${LIBRARY_PYTHON_BUILD}
    INSTALL_COMMAND ${LIBRARY_PYTHON_INSTALL}
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
