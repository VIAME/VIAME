# PyTorch External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

CreateDirectory( ${VIAME_BUILD_PREFIX}/src/pytorch-build )

set( PYTORCH_LIBS_TO_BUILD )

if( VIAME_PYTORCH_BUILD_FROM_SOURCE )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} pytorch )
endif()

if( VIAME_PYTORCH_BUILD_TORCHVISION )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} torchvision )
endif()

if( VIAME_ENABLE_PYTORCH-VIDEO )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} pyav torchvideo )
endif()

if( VIAME_ENABLE_PYTORCH-MMDET OR VIAME_ENABLE_PYTORCH-NETHARN )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} imgaug )
endif()

if( VIAME_ENABLE_PYTORCH-MMDET )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} mmcv mmdetection )
endif()

if( VIAME_ENABLE_ONNX AND VIAME_ENABLE_DARKNET)
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} darknet-to-pytorch-onnx )
endif()

if( VIAME_ENABLE_PYTORCH-MIT-YOLO )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} mit-yolo )
endif()

if( VIAME_ENABLE_ONNX AND VIAME_ENABLE_PYTORCH-MMDET )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} mmdeploy )
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

if( VIAME_ENABLE_PYTORCH-DETECTRON )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} detectron2 )
endif()

if( VIAME_ENABLE_PYTORCH-SAM )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} sam2 )
endif()

if( VIAME_ENABLE_PYTORCH-STEREO )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} foundation-stereo )
endif()

if( VIAME_ENABLE_PYTORCH-RF-DETR )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} rf-detr )
endif()

if( VIAME_ENABLE_TENSORRT )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} torch2rt )
endif()

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} ${PYTORCH_LIBS_TO_BUILD} )
set( PYTORCH_ENV_VARS ${PYTHON_DEP_ENV_VARS} )

if( VIAME_ENABLE_CUDNN )
  if( WIN32 )
    string( REPLACE ";" "----" CUDNN_ADJ_LIB_LIST "${CUDNN_LIBRARIES}" )
    list( APPEND PYTORCH_ENV_VARS "CUDNN_LIBRARY=${CUDNN_ADJ_LIB_LIST}" )
  else()
    list( APPEND PYTORCH_ENV_VARS "CUDNN_LIBRARY=${CUDNN_LIBRARIES}" )
  endif()
  if( CUDNN_ROOT_DIR )
    list( APPEND PYTORCH_ENV_VARS "CUDNN_INCLUDE_DIR=${CUDNN_ROOT_DIR}/include" )
  endif()
endif()

if( VIAME_PYTORCH_DISABLE_NINJA )
  list( APPEND PYTORCH_ENV_VARS "USE_NINJA=OFF" )
endif()

if( VIAME_ENABLE_PYTORCH-NETHARN )
  list( APPEND PYTORCH_ENV_VARS "BEZIER_NO_EXTENSION=1" )
endif()

if( VIAME_ENABLE_CUDA )
  list( APPEND PYTORCH_ENV_VARS "USE_CUDA=1" )
  list( APPEND PYTORCH_ENV_VARS "FORCE_CUDA=1" )
  list( APPEND PYTORCH_ENV_VARS "CUDA_VISIBLE_DEVICES=0" )
  list( APPEND PYTORCH_ENV_VARS "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}" )
  list( APPEND PYTORCH_ENV_VARS "CUDACXX=${CUDA_NVCC_EXECUTABLE}" )
  list( APPEND PYTORCH_ENV_VARS "TORCH_CUDA_ARCH_LIST=${CUDA_ARCHITECTURES}" )
  list( APPEND PYTORCH_ENV_VARS "TORCH_NVCC_FLAGS=-Xfatbin -compress-all" )
  list( APPEND PYTORCH_ENV_VARS "NO_CAFFE2_OPS=1" )
else()
  list( APPEND PYTORCH_ENV_VARS "USE_CUDA=0" )
endif()

if( VIAME_ENABLE_PYTORCH-MMDET )
  list( APPEND PYTORCH_ENV_VARS "MMCV_WITH_OPS=1" )
endif()

if( VIAME_PYTORCH_BUILD_TORCHVISION AND NOT WIN32 )
  list( APPEND PYTORCH_ENV_VARS "TORCHVISION_USE_PNG=0" )
endif()

if( WIN32 AND VIAME_ENABLE_LEARN AND Python_VERSION VERSION_GREATER "3.7" )
  list( APPEND PYTORCH_ENV_VARS "SETUPTOOLS_USE_DISTUTILS=1" )
endif()

foreach( LIB ${PYTORCH_LIBS_TO_BUILD} )
  if( "${LIB}" STREQUAL "pytorch" )
    set( LIBRARY_LOCATION ${VIAME_PACKAGES_DIR}/pytorch )
  elseif( "${LIB}" STREQUAL "roi-align" )
    set( LIBRARY_LOCATION ${VIAME_SOURCE_DIR}/plugins/pytorch/mdnet )
  elseif( "${LIB}" STREQUAL "pyav" )
    set( LIBRARY_LOCATION ${VIAME_PACKAGES_DIR}/python-utils/pyav )
  else()
    set( LIBRARY_LOCATION ${VIAME_PACKAGES_DIR}/pytorch-libs/${LIB} )
  endif()

  set( LIBRARY_LOCATION_URL file://${LIBRARY_LOCATION} )

  set( LIBRARY_PIP_CACHE_DIR ${VIAME_BUILD_PREFIX}/src/pytorch-build/${LIB}-cache )
  set( LIBRARY_PIP_BUILD_DIR ${VIAME_BUILD_PREFIX}/src/pytorch-build/${LIB}-build )
  set( LIBRARY_PIP_TMP_DIR ${VIAME_BUILD_PREFIX}/src/pytorch-build/${LIB}-tmp )

  CreateDirectory( ${LIBRARY_PIP_CACHE_DIR} )
  CreateDirectory( ${LIBRARY_PIP_BUILD_DIR} )
  CreateDirectory( ${LIBRARY_PIP_TMP_DIR} )

  set( LIBRARY_PIP_BUILD_DIR_CMD -b ${LIBRARY_PIP_BUILD_DIR} )
  set( LIBRARY_PIP_CACHE_DIR_CMD --cache-dir ${LIBRARY_PIP_CACHE_DIR} )

  # For each python library split the build and install into two steps.

  if( VIAME_PYTHON_SYMLINK )
    # In development mode, install with the -e flag for editable.
    set( LIBRARY_PIP_BUILD_CMD
      ${Python_EXECUTABLE} setup.py build )
    set( LIBRARY_PIP_INSTALL_CMD
      ${Python_EXECUTABLE} -m pip install --user -e . )
  else()
    # TODO:
    # replace direct calls to setup.py with `python -m build`
    if( "${LIB}" STREQUAL "mit-yolo" OR "${LIB}" STREQUAL "rf-detr" )
      # Use -m build for pyproject.toml-based packages
      # FIXME:
      # If we remove no-isolation then it will complain that pip cannot be found.
      # I don't know exactly why, but for now it works.
      set( LIBRARY_PIP_BUILD_CMD
        ${Python_EXECUTABLE} -m build
          --wheel
          --no-isolation
          --outdir ${LIBRARY_PIP_BUILD_DIR}
      )
    elseif( "${LIB}" STREQUAL "mmcv" OR "${LIB}" STREQUAL "torchvision" )
      set( LIBRARY_PIP_BUILD_CMD
        ${Python_EXECUTABLE} setup.py
          bdist_wheel -d ${LIBRARY_PIP_BUILD_DIR} )
    else()
      set( LIBRARY_PIP_BUILD_CMD
        ${Python_EXECUTABLE} setup.py build_ext
          --include-dirs="${VIAME_INSTALL_PREFIX}/include"
          --library-dirs="${VIAME_INSTALL_PREFIX}/lib"
          --inplace bdist_wheel -d ${LIBRARY_PIP_BUILD_DIR} )
    endif()
    set( LIBRARY_PIP_INSTALL_CMD
      ${CMAKE_COMMAND}
        -DPYTHON_EXECUTABLE=${Python_EXECUTABLE}
        -DPython_EXECUTABLE=${Python_EXECUTABLE}
        -DWHEEL_DIR=${LIBRARY_PIP_BUILD_DIR}
        -P ${VIAME_CMAKE_DIR}/install_python_wheel.cmake )
  endif()

  set( LIBRARY_PYTHON_BUILD
    ${CMAKE_COMMAND} -E env "${PYTORCH_ENV_VARS}"
    "TMPDIR=${LIBRARY_PIP_TMP_DIR}"
    ${LIBRARY_PIP_BUILD_CMD} )
  set( LIBRARY_PYTHON_INSTALL
    ${CMAKE_COMMAND} -E env "${PYTORCH_ENV_VARS}"
    "TMPDIR=${LIBRARY_PIP_TMP_DIR}"
    ${LIBRARY_PIP_INSTALL_CMD} )

  set( LIBRARY_PATCH_COMMAND "" )
  set( PROJECT_DEPS fletch python-deps )

  if( VIAME_ENABLE_SMQTK )
    set( PROJECT_DEPS ${PROJECT_DEPS} smqtk )
  endif()

  if( NOT "${LIB}" STREQUAL "pytorch" )
    set( PROJECT_DEPS ${PROJECT_DEPS} pytorch )
    if( VIAME_ENABLE_PYTORCH-VISION AND
        NOT "${LIB}" STREQUAL "torchvision" )
      set( PROJECT_DEPS ${PROJECT_DEPS} torchvision )
    endif()
  endif()

  if( "${LIB}" STREQUAL "bioharn" )
    set( PROJECT_DEPS netharn )
  elseif( "${LIB}" STREQUAL "netharn" )
    set( PROJECT_DEPS mmdetection )
  elseif( "${LIB}" STREQUAL "torch2rt" )
    set( PROJECT_DEPS fletch python-deps tensorrt )
  elseif( "${LIB}" STREQUAL "torchvision" )
    if( VIAME_PYTORCH_VERSION VERSION_LESS "1.11" )
      set( LIBRARY_PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${VIAME_PATCHES_DIR}/torchvision
        ${VIAME_PACKAGES_DIR}/pytorch-libs/torchvision )
    endif()
  elseif( "${LIB}" STREQUAL "detectron2" )
    if( VIAME_ENABLE_PYTORCH-NETHARN )
      set( PROJECT_DEPS ${PROJECT_DEPS} bioharn )
    endif()
    if( WIN32 )
      set( LIBRARY_PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${VIAME_PATCHES_DIR}/detectron2
        ${VIAME_PACKAGES_DIR}/pytorch-libs/detectron2 )
    endif()
  elseif( "${LIB}" STREQUAL "pyav" )
    set( LIBRARY_PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${VIAME_PATCHES_DIR}/pyav
      ${VIAME_PACKAGES_DIR}/python-utils/pyav )
  elseif( "${LIB}" STREQUAL "torchvideo" )
    set( PROJECT_DEPS ${PROJECT_DEPS} pyav )
  elseif( "${LIB}" STREQUAL "mmcv" )
    if( WIN32 )
      set( LIBRARY_PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${VIAME_PATCHES_DIR}/mmcv
        ${VIAME_PACKAGES_DIR}/pytorch-libs/mmcv )
    endif()
  elseif( "${LIB}" STREQUAL "mmdetection" )
    set( PROJECT_DEPS ${PROJECT_DEPS} mmcv )
  elseif( "${LIB}" STREQUAL "mmdeploy" )
    set( PROJECT_DEPS ${PROJECT_DEPS} mmdetection onnxruntimelibs )
  endif()

  # For slow-to-build packages, use conditional build that checks source hash
  # This prevents unnecessary recompilation when source hasn't changed
  set( SLOW_BUILD_PACKAGE FALSE )
  if( "${LIB}" STREQUAL "detectron2" OR "${LIB}" STREQUAL "sam2" OR "${LIB}" STREQUAL "mmdeploy" )
    set( SLOW_BUILD_PACKAGE TRUE )
    set( LIB_HASH_FILE ${VIAME_BUILD_PREFIX}/src/${LIB}-source-hash.txt )

    # Convert environment variables list to semicolon-separated string for passing to cmake script
    string( REPLACE ";" "----" PYTORCH_ENV_VARS_STR "${PYTORCH_ENV_VARS}" )
  endif()

  if( SLOW_BUILD_PACKAGE )
    # Use conditional build for slow packages (detectron2, sam2, mmdeploy)
    # This checks source hash and skips rebuild if unchanged

    # Base conditional build command
    set( CONDITIONAL_BUILD_CMD
      ${CMAKE_COMMAND}
        -DLIB_NAME=${LIB}
        -DLIB_SOURCE_DIR=${LIBRARY_LOCATION}
        -DHASH_FILE=${LIB_HASH_FILE}
        -DPYTHON_BUILD_CMD="${LIBRARY_PIP_BUILD_CMD}"
        -DENV_VARS="${PYTORCH_ENV_VARS_STR}"
        -DTMPDIR="${LIBRARY_PIP_TMP_DIR}"
        -DWORKING_DIR=${LIBRARY_LOCATION} )

    # mmdeploy has additional C++ build steps
    if( "${LIB}" STREQUAL "mmdeploy" )
      set( ONNXRUNTIME_DIR ${VIAME_PYTHON_PACKAGES}/onnxruntime/onnxruntimelibs )
      set( LIBRARY_CPP_BUILD_DIR ${VIAME_SOURCE_DIR}/packages/pytorch-libs/mmdeploy/build )
      file( MAKE_DIRECTORY ${LIBRARY_CPP_BUILD_DIR} )

      set( LIBRARY_CPP_CONFIG
        ${CMAKE_COMMAND}
        -DMMDEPLOY_TARGET_BACKENDS=ort
        -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}
        -S "${LIBRARY_LOCATION}"
        -B "${LIBRARY_CPP_BUILD_DIR}" )

      set( LIBRARY_CPP_BUILD ${CMAKE_COMMAND} --build "${LIBRARY_CPP_BUILD_DIR}" )
      set( LIBRARY_CPP_INSTALL ${CMAKE_COMMAND} --install "${LIBRARY_CPP_BUILD_DIR}" )
      if( (CMAKE_CONFIGURATION_TYPES STREQUAL "Release") OR (CMAKE_BUILD_TYPE STREQUAL "Release") )
        list( APPEND LIBRARY_CPP_BUILD --config Release )
        list( APPEND LIBRARY_CPP_INSTALL --config Release )
      endif()
      if( VIAME_BUILD_MAX_THREADS )
        list( APPEND LIBRARY_CPP_BUILD -j ${VIAME_BUILD_MAX_THREADS} )
      endif()

      list( APPEND CONDITIONAL_BUILD_CMD
        -DCPP_BUILD_CMD="${LIBRARY_CPP_BUILD}"
        -DCPP_INSTALL_CMD="${LIBRARY_CPP_INSTALL}" )

      set( LIBRARY_CONFIGURE_CMD ${LIBRARY_CPP_CONFIG} )
    else()
      set( LIBRARY_CONFIGURE_CMD "" )
    endif()

    list( APPEND CONDITIONAL_BUILD_CMD -P ${VIAME_CMAKE_DIR}/custom_build_python_dep.cmake )

    ExternalProject_Add( ${LIB}
      DEPENDS ${PROJECT_DEPS}
      PREFIX ${VIAME_BUILD_PREFIX}
      SOURCE_DIR ${LIBRARY_LOCATION}
      BUILD_IN_SOURCE 1
      PATCH_COMMAND ${LIBRARY_PATCH_COMMAND}
      CONFIGURE_COMMAND "${LIBRARY_CONFIGURE_CMD}"
      BUILD_COMMAND ${CONDITIONAL_BUILD_CMD}
      INSTALL_COMMAND ${LIBRARY_PYTHON_INSTALL}
      LIST_SEPARATOR "----" )

    if( "${LIB}" STREQUAL "mmdeploy" )
      set( MMDEPLOY_INSTALL_DIR ${VIAME_PYTHON_INSTALL}/site-packages/mmdeploy )
      ExternalProject_Add_Step(${LIB}
        postinstall
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${LIBRARY_LOCATION}/configs ${MMDEPLOY_INSTALL_DIR}/configs
        DEPENDEES install )
    endif()

  else()
    ExternalProject_Add( ${LIB}
      DEPENDS ${PROJECT_DEPS}
      PREFIX ${VIAME_BUILD_PREFIX}
      SOURCE_DIR ${LIBRARY_LOCATION}
      BUILD_IN_SOURCE 1
      PATCH_COMMAND ${LIBRARY_PATCH_COMMAND}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ${LIBRARY_PYTHON_BUILD}
      INSTALL_COMMAND ${LIBRARY_PYTHON_INSTALL}
      LIST_SEPARATOR "----"
      )
  endif()

  # For non-slow packages, use traditional forcebuild method if VIAME_FORCEBUILD is on
  if( NOT SLOW_BUILD_PACKAGE AND VIAME_FORCEBUILD )
    RemoveProjectCMakeStamp( ${LIB} )
  endif()
endforeach()

set( VIAME_ARGS_pytorch
  -Dpytorch_DIR:PATH=${VIAME_BUILD_PREFIX}/src/pytorch-build
  )
