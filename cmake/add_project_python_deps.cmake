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
set( VIAME_PYTHON_BASIC_DEPS "wheel" "ordered_set" "build" "cython" )

if( Python_VERSION VERSION_GREATER_EQUAL "3.12" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "numpy>=2.1.0,<=2.2.6" )
else()
  list( APPEND VIAME_PYTHON_BASIC_DEPS "numpy>=1.26.0,<=2.0.2" )
endif()

if( VIAME_BUILD_TESTS )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "pytest" )
endif()

# Setuptools version constraint
if( WIN32 AND VIAME_ENABLE_PYTORCH-NETHARN AND VIAME_ENABLE_GDAL )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "setuptools==57.5.0" )
else()
  list( APPEND VIAME_PYTHON_BASIC_DEPS "setuptools>=75.3.0" )
endif()

# For scoring and plotting
list( APPEND VIAME_PYTHON_BASIC_DEPS "kiwisolver" )
if( Python_VERSION VERSION_GREATER_EQUAL "3.12" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "matplotlib>=3.9.0,<=3.10.0" )
else()
  list( APPEND VIAME_PYTHON_BASIC_DEPS "matplotlib>=3.7.0,<=3.8.5" )
endif()

# Protobuf for siammask (if enabled)
if( VIAME_ENABLE_PYTORCH-SIAMMASK )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "protobuf" )
endif()

# For fusion classifier
list( APPEND VIAME_PYTHON_BASIC_DEPS "map_boxes" "ensemble_boxes" )

if( Python_VERSION VERSION_GREATER_EQUAL "3.12" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "llvmlite>=0.44.0" "numba>=0.61.0" )
else()
  list( APPEND VIAME_PYTHON_BASIC_DEPS "llvmlite>=0.43.0" "numba>=0.60.0" )
endif()

# For pytorch building
if( VIAME_BUILD_PYTORCH_FROM_SOURCE )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "typing-extensions" "bs4" )
endif()

# For mmdetection
if( VIAME_ENABLE_PYTORCH-MMDET )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "yapf" )
endif()

# For measurement scripts
if( VIAME_ENABLE_OPENCV )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "tqdm" "scipy" )
endif()

# For PostgreSQL database support (used by native ITQ indexer, etc.)
if( VIAME_ENABLE_POSTGRESQL )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "psycopg2-binary" )
endif()

# For ONNX
if( VIAME_ENABLE_ONNX )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "onnx<=1.16.1" )
endif()

# For LEARN models
if( VIAME_ENABLE_PYTORCH-LEARN )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "wandb" "fsspec" "pyarrow" "filelock" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "timm" "submitit" "scikit-learn" )
endif()

if( VIAME_ENABLE_KEYPOINT )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "msgpack" )
endif()

if( VIAME_ENABLE_PYTORCH )
  if( Python_VERSION VERSION_GREATER_EQUAL "3.12" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "scikit-image>=0.24.0" )
  else()
    list( APPEND VIAME_PYTHON_BASIC_DEPS "scikit-image>=0.22.0,<=0.24.0" )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH-MMDET OR VIAME_ENABLE_PYTORCH-NETHARN )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "scikit-build" "async_generator" )
endif()

if( VIAME_ENABLE_PYTORCH-NETHARN )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "six" "scriptconfig" "parse" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "kwarray" "kwimage" "kwplot" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "torch_liberator" "liberator" )
endif()

if( VIAME_ENABLE_OPENCV OR VIAME_ENABLE_PYTORCH-NETHARN OR
    VIAME_ENABLE_PYTORCH-MIT-YOLO OR VIAME_ENABLE_PYTORCH-ULTRALYTICS )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "ubelt" "ndsampler" "pygments" "kwutil" )

  if( Python_VERSION VERSION_GREATER_EQUAL "3.12" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "networkx>=3.4" "pandas>=2.2.0" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "imageio>=2.36.0" "kwcoco>=0.8.5" "colormath" )
  else()
    list( APPEND VIAME_PYTHON_BASIC_DEPS "networkx>=3.2,<=3.4" "pandas>=2.1.0,<=2.2.3" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "imageio>=2.34.0" "kwcoco>=0.8.0" "colormath" )
  endif()

  if( VIAME_ENABLE_PYTORCH-ULTRALYTICS )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "seaborn>=0.13.2" )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH-HUGGINGFACE )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "transformers>=4.49.0" )
endif()

if( VIAME_ENABLE_PYTORCH AND VIAME_BUILD_PYTORCH_FROM_SOURCE )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "pyyaml" )
endif()

if( VIAME_ENABLE_PYTORCH AND VIAME_ENABLE_PYTORCH-MMDET )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "pycocotools" )
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

list( REMOVE_DUPLICATES VIAME_PYTHON_BASIC_DEPS )

# ------------------------------ ADD ANY ADV PYTHON DEPS HERE ------------------------------------
# Advanced python dependencies are installed individually due to special reqs

list( APPEND VIAME_PYTHON_ADV_DEPS python-deps )
set( VIAME_PYTHON_ADV_DEP_CMDS "custom-install" )

if( VIAME_ENABLE_KEYPOINT )
  list( APPEND VIAME_PYTHON_ADV_DEPS wxPython )

  # wxPython 4.2.2+ required for Python 3.12/3.13 support
  if( Python_VERSION VERSION_GREATER_EQUAL "3.12" )
    set( WX_VERSION "4.2.2" )
  else()
    set( WX_VERSION "4.2.1" )
  endif()

  list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "wxPython>=${WX_VERSION}" )
endif()

if( VIAME_ENABLE_PYTORCH AND
    ( NOT VIAME_BUILD_PYTORCH_FROM_SOURCE OR
      ( VIAME_ENABLE_PYTORCH-VISION AND NOT VIAME_BUILD_TORCHVISION_FROM_SOURCE ) ) )

  if( VIAME_ENABLE_CUDA )
    set( TORCH_CUDA_VER_STR "cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}" )
  else()
    set( TORCH_CUDA_VER_STR "cpu" )
  endif()

  set( PYTORCH_ARCHIVE "https://download.pytorch.org/whl/${TORCH_CUDA_VER_STR}" )
  set( TORCH_URL_CMD "--extra-index-url ${PYTORCH_ARCHIVE}" )

  if( NOT VIAME_BUILD_PYTORCH_FROM_SOURCE )
    set( PYTORCH_CMD "torch==${VIAME_PYTORCH_VERSION}" )

    list( APPEND VIAME_PYTHON_ADV_DEPS pytorch )
    list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "${PYTORCH_CMD} ${TORCH_URL_CMD}" )
  endif()

  if( VIAME_ENABLE_PYTORCH-VISION AND NOT VIAME_BUILD_TORCHVISION_FROM_SOURCE )
    if( VIAME_PYTORCH_VERSION VERSION_EQUAL "2.7.0" )
      set( TORCHVISION_CMD "torchvision==0.22.0" )
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

  # Build the full command list and convert to ----separated string
  set( PYTHON_DEP_FULL_CMD ${Python_EXECUTABLE} -m ${PYTHON_DEP_PIP_CMD} )
  string( REPLACE ";" "----" PYTHON_DEP_CMD_STR "${PYTHON_DEP_FULL_CMD}" )
  string( REPLACE ";" "----" PYTHON_DEP_ENV_STR "${PYTHON_DEP_ENV_VARS}" )

  set( PYTHON_DEP_BUILD
    ${CMAKE_COMMAND}
      -DCOMMAND_TO_RUN:STRING=${PYTHON_DEP_CMD_STR}
      -DENV_VARS:STRING=${PYTHON_DEP_ENV_STR}
      -P ${VIAME_CMAKE_DIR}/run_python_command.cmake
    )

  if( "${DEP}" STREQUAL "pytorch" )
    set( PYTHON_DEP_INSTALL ${CMAKE_COMMAND}
      -DVIAME_PYTHON_BASE:PATH=${VIAME_PYTHON_INSTALL}
      -DVIAME_PATCH_DIR:PATH=${VIAME_SOURCE_DIR}/packages/patches
      -DVIAME_PYTORCH_VERSION:STRING=${VIAME_PYTORCH_VERSION}
      -P ${VIAME_SOURCE_DIR}/cmake/custom_install_pytorch.cmake )
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

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} pymotmetrics )

set( PROJECT_DEPS fletch python-deps )
set( LIBRARY_LOCATION ${VIAME_PACKAGES_DIR}/python-utils/pymotmetrics )
set( LIBRARY_PIP_BUILD_DIR ${VIAME_BUILD_PREFIX}/src/pymotmetrics-build )
CreateDirectory( ${LIBRARY_PIP_BUILD_DIR} )

if( VIAME_PYTHON_SYMLINK )
  set( LIBRARY_PIP_BUILD_CMD
    ${Python_EXECUTABLE} setup.py build --build-base=${LIBRARY_PIP_BUILD_DIR} )
  set( LIBRARY_PIP_INSTALL_CMD
    ${Python_EXECUTABLE} -m pip install --user -e . )
else()
  set( LIBRARY_PIP_BUILD_CMD
    ${Python_EXECUTABLE} setup.py
      build --build-base=${LIBRARY_PIP_BUILD_DIR}
      bdist_wheel -d ${LIBRARY_PIP_BUILD_DIR} )
  set( LIBRARY_PIP_INSTALL_CMD
    ${CMAKE_COMMAND}
      -DPYTHON_EXECUTABLE=${Python_EXECUTABLE}
      -DPython_EXECUTABLE=${Python_EXECUTABLE}
      -DWHEEL_DIR=${LIBRARY_PIP_BUILD_DIR}
      -P ${VIAME_CMAKE_DIR}/install_python_wheel.cmake )
endif()

# Convert commands and env vars to ----separated strings for the wrapper script
string( REPLACE ";" "----" LIBRARY_BUILD_CMD_STR "${LIBRARY_PIP_BUILD_CMD}" )
string( REPLACE ";" "----" LIBRARY_INSTALL_CMD_STR "${LIBRARY_PIP_INSTALL_CMD}" )
string( REPLACE ";" "----" PYMOT_ENV_STR "${PYTHON_DEP_ENV_VARS}" )

set( LIBRARY_PYTHON_BUILD
  ${CMAKE_COMMAND}
    -DCOMMAND_TO_RUN:STRING=${LIBRARY_BUILD_CMD_STR}
    -DENV_VARS:STRING=${PYMOT_ENV_STR}
    -DWORKING_DIR:PATH=${LIBRARY_LOCATION}
    -P ${VIAME_CMAKE_DIR}/run_python_command.cmake )
set( LIBRARY_PYTHON_INSTALL
  ${CMAKE_COMMAND}
    -DCOMMAND_TO_RUN:STRING=${LIBRARY_INSTALL_CMD_STR}
    -DENV_VARS:STRING=${PYMOT_ENV_STR}
    -DWORKING_DIR:PATH=${LIBRARY_LOCATION}
    -P ${VIAME_CMAKE_DIR}/run_python_command.cmake )

ExternalProject_Add( pymotmetrics
  DEPENDS ${PROJECT_DEPS}
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${LIBRARY_LOCATION}
  BUILD_IN_SOURCE 1
  USES_TERMINAL_BUILD 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${LIBRARY_PYTHON_BUILD}
  INSTALL_COMMAND ${LIBRARY_PYTHON_INSTALL}
  LIST_SEPARATOR "----" )

