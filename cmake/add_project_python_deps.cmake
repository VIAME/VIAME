# ------------------------------------------------------------------------------------------------
# Python Dependency External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

# ------------------------------ ADD ANY BASIC PYTHON DEPS HERE ----------------------------------
# Basic dependencies are installed jointly in one local pip installation call

# Core requirements used for building certain libraries
set( VIAME_PYTHON_BASIC_DEPS "wheel" "ordered_set" "cython<3.0.0" )

if( VIAME_FIXUP_BUNDLE AND Python_VERSION VERSION_LESS "3.8" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "numpy==1.19.3" )
else()
  list( APPEND  VIAME_PYTHON_BASIC_DEPS "numpy<=1.25.2" )
endif()

if( VIAME_BUILD_TESTS )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "pytest" )
endif()

# Setuptools < 58.0 required for current version of gdal on windows or earlier python
if( ( WIN32 OR Python_VERSION VERSION_LESS "3.8" )
    AND VIAME_ENABLE_PYTORCH-NETHARN
    AND VIAME_ENABLE_GDAL )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "setuptools==57.5.0" )
else()
  # 75.3.0 is the last version to support Python 3.8
  list( APPEND VIAME_PYTHON_BASIC_DEPS "setuptools==75.3.0" )
endif()

# For scoring and plotting
list( APPEND VIAME_PYTHON_BASIC_DEPS "kiwisolver<=1.4.7" )
list( APPEND VIAME_PYTHON_BASIC_DEPS "matplotlib<=3.6.2" )

# For netharn and mmdet de-pickle on older versions
if( Python_VERSION VERSION_LESS "3.8" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "pickle5" )

  if( VIAME_ENABLE_PYTORCH-PYSOT )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "protobuf==3.19.4" )
  endif()
endif()

# For fusion classifier
list( APPEND VIAME_PYTHON_BASIC_DEPS "map_boxes" "ensemble_boxes" )

if( Python_VERSION VERSION_LESS "3.9" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "llvmlite==0.31.0" "numba==0.47" )
else()
  list( APPEND VIAME_PYTHON_BASIC_DEPS "llvmlite==0.43.0" "numba==0.60" )
endif()

# For pytorch building
if( VIAME_PYTORCH_BUILD_FROM_SOURCE )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "typing-extensions" "bs4" )
endif()

# For mmdetection
if( VIAME_ENABLE_PYTORCH-MMDET )
  if( Python_VERSION VERSION_LESS "3.8" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "yapf<=0.32.0" )
  else()
    list( APPEND VIAME_PYTHON_BASIC_DEPS "yapf" )
  endif()
endif()

# For measurement scripts
if( VIAME_ENABLE_OPENCV )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "tqdm" "scipy" )
endif()

if( VIAME_ENABLE_LEARN AND Python_VERSION VERSION_LESS "3.7" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "wandb<=0.15.7" "fsspec<=2022.1.0" )
  if( WIN32 )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "pyarrow==4.0.0" )
  else()
    list( APPEND VIAME_PYTHON_BASIC_DEPS "pyarrow==2.0.0" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "filelock==3.4.1" )
  endif()
endif()

if( ( WIN32 OR NOT VIAME_ENABLE_OPENCV ) AND
      ( VIAME_ENABLE_OPENCV OR
        VIAME_ENABLE_PYTORCH-MMDET OR
        VIAME_ENABLE_PYTORCH-NETHARN ) )
  if( WIN32 AND NOT VIAME_ENABLE_WIN32GUI )
    if( Python_VERSION VERSION_LESS "3.7" )
      list( APPEND VIAME_PYTHON_BASIC_DEPS "opencv-python-headless<=4.6.0.66" )
    else()
      list( APPEND VIAME_PYTHON_BASIC_DEPS "opencv-python-headless" )
    endif()
  else()
    if( Python_VERSION VERSION_LESS "3.7" )
      list( APPEND VIAME_PYTHON_BASIC_DEPS "opencv-python<=4.6.0.66" )
    else()
      list( APPEND VIAME_PYTHON_BASIC_DEPS "opencv-python" )
    endif()
  endif()
endif()

if( VIAME_ENABLE_KEYPOINT )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "msgpack" )
endif()

if( VIAME_ENABLE_PYTORCH )
  if( Python_VERSION VERSION_LESS "3.10" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "scikit-image==0.16.2" )
  else()
    list( APPEND VIAME_PYTHON_BASIC_DEPS "scikit-image==0.24.0" )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH-MMDET OR VIAME_ENABLE_PYTORCH-NETHARN )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "scikit-build" "async_generator" )
endif()

if( VIAME_ENABLE_PYTORCH-NETHARN AND VIAME_ENABLE_GDAL )
  if( WIN32 AND Python_VERSION VERSION_LESS "3.8" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "gdal==2.2.3" )
  else()
    list( APPEND VIAME_PYTHON_BASIC_DEPS "gdal" )
  endif()
endif()

if( VIAME_ENABLE_OPENCV OR VIAME_ENABLE_PYTORCH-NETHARN )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "ubelt<=1.3.7" "ndsampler<=0.8.0" "pygments" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "networkx<=2.8.8" "pandas<=1.5.3" )

  if( Python_VERSION VERSION_LESS "3.10" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "imageio==2.15.0" "kwcoco==0.2.31" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "bezier==2020.1.14" )
  else()
    list( APPEND VIAME_PYTHON_BASIC_DEPS "imageio==2.36.0" "kwcoco==0.8.5" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "colormath" )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH-ULTRALYTICS )
  # Ultralytics will be installed as an advanced package. Here we need to
  # explicitly install the dependencies of the package and our wrapper.
  list( APPEND VIAME_PYTHON_BASIC_DEPS "seaborn==0.13.2" )  # can likely keep this version loose
  list( APPEND VIAME_PYTHON_BASIC_DEPS "ubelt<=1.3.7" )
  if( Python_VERSION VERSION_LESS "3.10" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "kwcoco==0.2.31" )
  else()
    list( APPEND VIAME_PYTHON_BASIC_DEPS "kwcoco==0.8.5" )
  endif()
endif()

if( Python_VERSION VERSION_LESS "3.7" AND
    ( VIAME_ENABLE_PYTORCH-PYSOT OR VIAME_ENABLE_SMQTK ) )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "tensorboardX<=2.6.0" )
endif()

if( VIAME_ENABLE_PYTORCH AND VIAME_PYTORCH_BUILD_FROM_SOURCE )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "pyyaml" )
endif()

if( VIAME_ENABLE_PYTORCH AND VIAME_ENABLE_PYTORCH-MMDET )
  if( WIN32 AND Python_VERSION VERSION_LESS "3.7" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "pycocotools-windows" )
  else()
    list( APPEND VIAME_PYTHON_BASIC_DEPS "pycocotools" )
  endif()
endif()

if( VIAME_PYTHON_BUILD_FROM_SOURCE AND UNIX )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "backports.lzma" "backports.weakref" )
endif()

if( VIAME_ENABLE_TENSORFLOW )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "humanfriendly" )
  set( TF_ARGS "" )

  if( VIAME_TENSORFLOW_VERSION VERSION_LESS "2.0" )
    if( VIAME_ENABLE_CUDA )
      set( TF_ARGS "-gpu" )
    endif()
  else()
    if( NOT VIAME_ENABLE_CUDA )
      set( TF_ARGS "-cpu" )
    endif()
  endif()

  set( TF_ARGS "${TF_ARGS}==${VIAME_TENSORFLOW_VERSION}" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "tensorflow${TF_ARGS}" )
endif()

list(REMOVE_DUPLICATES VIAME_PYTHON_BASIC_DEPS)

# ------------------------------ ADD ANY ADV PYTHON DEPS HERE ------------------------------------
# Advanced python dependencies are installed individually due to special reqs

list( APPEND VIAME_PYTHON_ADV_DEPS python-deps )
set( VIAME_PYTHON_ADV_DEP_CMDS "custom-install" )

if( VIAME_ENABLE_KEYPOINT )
  set( WX_VERSION "4.0.7" )

  list( APPEND VIAME_PYTHON_ADV_DEPS wxPython )

  if( UNIX )
    if( EXISTS "/etc/os-release" )
      ParseLinuxOSField( "ID" OS_ID )
    endif()

    execute_process( COMMAND lsb_release -cs
      OUTPUT_VARIABLE RELEASE_CODENAME
      OUTPUT_STRIP_TRAILING_WHITESPACE )

    if( "${OS_ID}" MATCHES "centos" )
      set( WXP_ARCHIVE https://extras.wxpython.org/wxPython4/extras/linux/gtk3/centos-7 )
      list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "-U -f ${WXP_ARCHIVE} wxPython==${WX_VERSION}" )
    elseif( "${RELEASE_CODENAME}" MATCHES "xenial" )
      set( WXP_ARCHIVE https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04 )
      list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "-U -f ${WXP_ARCHIVE} wxPython==${WX_VERSION}" )
    else()
      list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "wxPython==${WX_VERSION}" )
    endif()
  else()
    list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "wxPython==${WX_VERSION}" )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH AND
    ( NOT VIAME_PYTORCH_BUILD_FROM_SOURCE OR
      ( VIAME_ENABLE_PYTORCH-VISION AND NOT VIAME_PYTORCH_BUILD_TORCHVISION ) ) )

  if( VIAME_ENABLE_CUDA )
    set( TORCH_CUDA_VER_STR "cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}" )
  else()
    set( TORCH_CUDA_VER_STR "cpu" )
  endif()

  set( PYTORCH_ARCHIVE "https://download.pytorch.org/whl/${TORCH_CUDA_VER_STR}" )
  set( TORCH_URL_CMD "--extra-index-url ${PYTORCH_ARCHIVE}" )

  if( NOT VIAME_PYTORCH_BUILD_FROM_SOURCE )
    set( PYTORCH_CMD "torch==${VIAME_PYTORCH_VERSION}" )

    list( APPEND VIAME_PYTHON_ADV_DEPS pytorch )
    list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "${PYTORCH_CMD} ${TORCH_URL_CMD}" )
  endif()

  if( VIAME_ENABLE_PYTORCH-VISION AND NOT VIAME_PYTORCH_BUILD_TORCHVISION )
    if( VIAME_PYTORCH_VERSION VERSION_EQUAL "2.5.1" )
      set( TORCHVISION_CMD "torchvision==0.20.1" )
    elseif( VIAME_PYTORCH_VERSION VERSION_EQUAL "1.13.1" )
      set( TORCHVISION_CMD "torchvision==0.13.0" )
    else()
      message( FATAL_ERROR "Unknown PyTorch version, unable to select the "
        "corresponding TorchVision wheel to use" )
    endif()

    list( APPEND VIAME_PYTHON_ADV_DEPS torchvision )
    list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "${TORCHVISION_CMD} ${TORCH_URL_CMD}" )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH-ULTRALYTICS )
  # Add ultralytics as an advanced dependency to avoid installing its strict
  # dependencies that are not needed.
  list( APPEND VIAME_PYTHON_ADV_DEPS ultralytics )
  list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "ultralytics<=8.3.71 ultralytics_thop==2.0.14 --no-deps" )
endif()

# ------------------------------------- INSTALL ROUTINES -----------------------------------------

set( VIAME_PYTHON_DEPS_DEPS fletch )

list( LENGTH VIAME_PYTHON_ADV_DEPS DEP_COUNT )
math( EXPR DEP_COUNT "${DEP_COUNT} - 1" )


# Force a pip update before installing other packages
set( PYTHON_DEP_BUILD
  ${CMAKE_COMMAND} -E env "${PYTHON_DEP_ENV_VARS}"
    ${Python_EXECUTABLE} -m pip install --user pip==25.0
  )
ExternalProject_Add( python-pip
  DEPENDS ${VIAME_PYTHON_DEPS_DEPS}
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_CMAKE_DIR}
  USES_TERMINAL_BUILD 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${PYTHON_DEP_BUILD}
  INSTALL_COMMAND ""
  INSTALL_DIR ${VIAME_INSTALL_PREFIX}
  LIST_SEPARATOR "----"
  )

list( APPEND VIAME_PYTHON_DEPS_DEPS python-pip )

foreach( ID RANGE ${DEP_COUNT} )

  list( GET VIAME_PYTHON_ADV_DEPS ${ID} DEP )

  set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} ${DEP} )

  if( "${DEP}" STREQUAL "python-deps" )
    set( PYTHON_LIB_DEPS ${VIAME_PYTHON_DEPS_DEPS} )
    set( CMD ${VIAME_PYTHON_BASIC_DEPS} )
  else()
    set( PYTHON_LIB_DEPS ${VIAME_PYTHON_DEPS_DEPS} python-deps )
    list( GET VIAME_PYTHON_ADV_DEP_CMDS ${ID} CMD )
  endif()

  set( PYTHON_DEP_PIP_CMD pip install --user ${CMD} )
  string( REPLACE " " ";" PYTHON_DEP_PIP_CMD "${PYTHON_DEP_PIP_CMD}" )

  set( PYTHON_DEP_BUILD
    ${CMAKE_COMMAND} -E env "${PYTHON_DEP_ENV_VARS}"
      ${Python_EXECUTABLE} -m ${PYTHON_DEP_PIP_CMD}
    )

  if( "${DEP}" STREQUAL "pytorch" )
    set( PYTHON_DEP_INSTALL ${CMAKE_COMMAND}
      -DVIAME_PYTHON_BASE:PATH=${VIAME_PYTHON_INSTALL}
      -DVIAME_PATCH_DIR:PATH=${VIAME_SOURCE_DIR}/packages/patches
      -DVIAME_PYTORCH_VERSION:STRING=${VIAME_PYTORCH_VERSION}
      -P ${VIAME_SOURCE_DIR}/cmake/custom_install_pytorch.cmake )
  elseif( "${DEP}" STREQUAL "python-deps" )
    set( PYTHON_DEP_INSTALL ${CMAKE_COMMAND}
      -DVIAME_PYTHON_BASE:PATH=${VIAME_PYTHON_INSTALL}
      -DVIAME_PATCH_DIR:PATH=${VIAME_SOURCE_DIR}/packages/patches
      -DVIAME_PYTHON_VERSION:STRING=${Python_VERSION}
      -P ${VIAME_SOURCE_DIR}/cmake/custom_install_python_deps.cmake )
  else()
    set( PYTHON_DEP_INSTALL "" )
  endif()

  ExternalProject_Add( ${DEP}
    DEPENDS ${PYTHON_LIB_DEPS}
    PREFIX ${VIAME_BUILD_PREFIX}
    SOURCE_DIR ${VIAME_CMAKE_DIR}
    USES_TERMINAL_BUILD 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${PYTHON_DEP_BUILD}
    INSTALL_COMMAND "${PYTHON_DEP_INSTALL}"
    INSTALL_DIR ${VIAME_INSTALL_PREFIX}
    LIST_SEPARATOR "----"
    )
endforeach()

# ------------------------------------- PYMOTMETRICS -----------------------------------------

# TODO: refactor this and make more generic for any python-utils requiring custom whl builds

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} pymotmetrics )

set( PROJECT_DEPS fletch python-deps )

if( VIAME_PYTHON_SYMLINK )
  set( LIBRARY_PIP_BUILD_CMD
    ${Python_EXECUTABLE} setup.py build )
  set( LIBRARY_PIP_INSTALL_CMD
    ${Python_EXECUTABLE} -m pip install --user -e . )
else()
  set( LIBRARY_PIP_BUILD_DIR ${VIAME_BUILD_PREFIX}/src/pymotmetrics-build )
  CreateDirectory( ${LIBRARY_PIP_BUILD_DIR} )

  set( LIBRARY_PIP_BUILD_CMD
    ${Python_EXECUTABLE} setup.py build_ext
      --include-dirs="${VIAME_INSTALL_PREFIX}/include"
      --library-dirs="${VIAME_INSTALL_PREFIX}/lib"
      --inplace bdist_wheel -d ${LIBRARY_PIP_BUILD_DIR} )
  set( LIBRARY_PIP_INSTALL_CMD
    ${CMAKE_COMMAND}
      -DPYTHON_EXECUTABLE=${Python_EXECUTABLE}
      -DPython_EXECUTABLE=${Python_EXECUTABLE}
      -DWHEEL_DIR=${LIBRARY_PIP_BUILD_DIR}
      -P ${VIAME_CMAKE_DIR}/install_python_wheel.cmake )
endif()

set( LIBRARY_PYTHON_BUILD
  ${CMAKE_COMMAND} -E env "${PYTHON_DEP_ENV_VARS}"
  ${LIBRARY_PIP_BUILD_CMD} )
set( LIBRARY_PYTHON_INSTALL
  ${CMAKE_COMMAND} -E env "${PYTHON_DEP_ENV_VARS}"
  ${LIBRARY_PIP_INSTALL_CMD} )

ExternalProject_Add( pymotmetrics
  DEPENDS ${PROJECT_DEPS}
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/python-utils/pymotmetrics
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${LIBRARY_PYTHON_BUILD}
  INSTALL_COMMAND ${LIBRARY_PYTHON_INSTALL}
  LIST_SEPARATOR "----"
  )

