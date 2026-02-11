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
set( VIAME_PYTHON_BASIC_DEPS "ordered_set" "build" "cython" )

# Setuptools and wheel version constraints
if( WIN32 AND VIAME_ENABLE_PYTORCH-NETHARN AND VIAME_ENABLE_GDAL )
  # Legacy constraint for GDAL compatibility - use older wheel that works with old setuptools
  list( APPEND VIAME_PYTHON_BASIC_DEPS "setuptools==57.5.0" "wheel<0.45.0" )
else()
  # Modern versions - wheel 0.45+ works with setuptools 70.1+
  list( APPEND VIAME_PYTHON_BASIC_DEPS "setuptools>=75.3.0" "wheel>=0.45.0" )
endif()

# Numpy versioning
if( Python_VERSION VERSION_GREATER_EQUAL "3.12" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "numpy>=2.1.0,<=2.2.6" )
else()
  list( APPEND VIAME_PYTHON_BASIC_DEPS "numpy>=1.26.0,<=2.0.2" )
endif()

# Testing infrastructure
if( VIAME_ENABLE_TESTS )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "pytest" )
endif()

# For KWIVER v2.0
if( VIAME_ENABLE_KWIVER )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "pygccxml" "castxml" )
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
if( VIAME_ENABLE_OPENCV OR VIAME_ENABLE_PYTORCH-SAM3 )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "tqdm" "scipy" )
endif()

# For PostgreSQL database support (used by native ITQ indexer, etc.)
if( VIAME_ENABLE_POSTGRESQL )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "psycopg2-binary" )
endif()

# For LEARN models
if( VIAME_ENABLE_PYTORCH-LEARN )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "wandb" "fsspec" "filelock" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "submitit" "scikit-learn" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "scipy" "termcolor" "addict" "yapf" )
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
  list( APPEND VIAME_PYTHON_BASIC_DEPS "scikit-build" )
endif()

if( VIAME_ENABLE_PYTORCH-NETHARN )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "scriptconfig" "parse" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "kwarray" "kwimage" "kwplot" )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "astunparse" "pygtrie" "pyflakes" )
endif()

if( VIAME_ENABLE_OPENCV OR VIAME_ENABLE_PYTORCH-NETHARN OR
    VIAME_ENABLE_PYTORCH-MIT-YOLO OR VIAME_ENABLE_PYTORCH-ULTRALYTICS )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "ubelt" "ndsampler" "pygments" "kwutil" )

  if( Python_VERSION VERSION_GREATER_EQUAL "3.12" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "networkx>=3.4" "pandas>=2.2.0" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "imageio>=2.36.0" )
  else()
    list( APPEND VIAME_PYTHON_BASIC_DEPS "networkx>=3.2,<=3.4" "pandas>=2.1.0,<=2.2.3" )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "imageio>=2.34.0" )
  endif()

  if( VIAME_ENABLE_PYTORCH-ULTRALYTICS )
    list( APPEND VIAME_PYTHON_BASIC_DEPS "seaborn>=0.13.2" "py-cpuinfo" )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH-MIT-YOLO )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "ruamel.yaml" )
endif()

if( VIAME_ENABLE_PYTORCH AND VIAME_BUILD_PYTORCH_FROM_SOURCE )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "pyyaml" )
endif()

if( VIAME_ENABLE_PYTORCH AND NOT VIAME_BUILD_PYTORCH_FROM_SOURCE )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "torch==${VIAME_PYTORCH_VERSION}" )
endif()

if( VIAME_ENABLE_PYTORCH-HUGGINGFACE OR VIAME_ENABLE_PYTORCH-RF-DETR )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "transformers>=4.49.0,<5.0.0" )
endif()

if( VIAME_ENABLE_PYTORCH-LEARN OR
    VIAME_ENABLE_PYTORCH-DETECTRON2 OR
    VIAME_ENABLE_PYTORCH-SAM3 OR
    VIAME_ENABLE_PYTORCH-STEREO )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "huggingface_hub" "safetensors" )
endif()

if( VIAME_ENABLE_ONNX )
  list( APPEND VIAME_PYTHON_BASIC_DEPS "onnx<=1.16.1" )
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

# ------------------------------ TORCH-DEPENDENT PIP PACKAGES -------------------------------------
# These packages require torch to be installed first (installed in add_project_pytorch.cmake)

set( VIAME_PYTHON_DEPS_REQ_TORCH "" )

if( VIAME_ENABLE_PYTORCH-NETHARN )
  list( APPEND VIAME_PYTHON_DEPS_REQ_TORCH "torch_liberator" "liberator"
    "networkx-algo-common-subtree>=0.2.0" "colormath" )
endif()

if( VIAME_ENABLE_PYTORCH-ULTRALYTICS )
  list( APPEND VIAME_PYTHON_DEPS_REQ_TORCH "ultralytics<=8.3.71" )
  list( APPEND VIAME_PYTHON_DEPS_REQ_TORCH "ultralytics_thop==2.0.14" )
endif()

if( VIAME_ENABLE_OPENCV OR VIAME_ENABLE_PYTORCH-NETHARN OR
    VIAME_ENABLE_PYTORCH-MIT-YOLO OR VIAME_ENABLE_PYTORCH-ULTRALYTICS )
  if( Python_VERSION VERSION_GREATER_EQUAL "3.12" )
    list( APPEND VIAME_PYTHON_DEPS_REQ_TORCH "kwcoco>=0.8.5" )
  else()
    list( APPEND VIAME_PYTHON_DEPS_REQ_TORCH "kwcoco>=0.8.0" )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH-LEARN OR
    VIAME_ENABLE_PYTORCH-DETECTRON2 OR
    VIAME_ENABLE_PYTORCH-SAM3 OR
    VIAME_ENABLE_PYTORCH-STEREO )
  list( APPEND VIAME_PYTHON_DEPS_REQ_TORCH "timm" )
endif()

if( VIAME_ENABLE_PYTORCH-RF-DETR )
  list( APPEND VIAME_PYTHON_DEPS_REQ_TORCH "supervision" "defusedxml>=0.7.1" "pyDeprecate" )
endif()

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

if( VIAME_ENABLE_PYTORCH )
  if( VIAME_ENABLE_CUDA )
    set( TORCH_CUDA_VER_STR "cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}" )
  else()
    set( TORCH_CUDA_VER_STR "cpu" )
  endif()
  set( PYTORCH_ARCHIVE "https://download.pytorch.org/whl/${TORCH_CUDA_VER_STR}" )
endif()

if( VIAME_ENABLE_PYTORCH AND
    ( NOT VIAME_BUILD_PYTORCH_FROM_SOURCE OR
      ( VIAME_ENABLE_PYTORCH-VISION AND NOT VIAME_BUILD_TORCHVISION_FROM_SOURCE ) ) )

  set( TORCH_URL_CMD "--extra-index-url ${PYTORCH_ARCHIVE}" )

  if( NOT VIAME_BUILD_PYTORCH_FROM_SOURCE )
    set( PYTORCH_CMD "torch==${VIAME_PYTORCH_VERSION}" )

    list( APPEND VIAME_PYTHON_ADV_DEPS pytorch )
    list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "${PYTORCH_CMD} ${TORCH_URL_CMD}" )
  endif()

  if( VIAME_ENABLE_PYTORCH-VISION AND NOT VIAME_BUILD_TORCHVISION_FROM_SOURCE )
    if( VIAME_PYTORCH_VERSION VERSION_EQUAL "2.10.0" )
      set( TORCHVISION_CMD "torchvision==0.25.0" )
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

# pymotmetrics from source - currently always enabled
list( APPEND VIAME_PYTHON_ADV_DEPS pymotmetrics )
if( VIAME_PYTHON_SYMLINK )
  list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "-e ${VIAME_PACKAGES_DIR}/python-utils/pymotmetrics" )
else()
  list( APPEND VIAME_PYTHON_ADV_DEP_CMDS "${VIAME_PACKAGES_DIR}/python-utils/pymotmetrics" )
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
    # Add PyTorch extra index URL for basic deps that include torch
    if( VIAME_ENABLE_PYTORCH AND NOT VIAME_BUILD_PYTORCH_FROM_SOURCE )
      set( PYTHON_DEP_EXTRA_INDEX "--extra-index-url" "${PYTORCH_ARCHIVE}" )
    else()
      set( PYTHON_DEP_EXTRA_INDEX "" )
    endif()
  else()
    set( PYTHON_LIB_DEPS ${VIAME_PYTHON_DEPS_DEPS} python-deps )
    list( GET VIAME_PYTHON_ADV_DEP_CMDS ${ID} CMD )
    set( PYTHON_DEP_EXTRA_INDEX "" )
  endif()

  set( PYTHON_DEP_PIP_CMD pip install --user ${PYTHON_DEP_EXTRA_INDEX} ${CMD} )
  if( VIAME_BUILD_NO_CACHE_DIR )
    set( PYTHON_DEP_PIP_CMD pip install --user --no-cache-dir ${PYTHON_DEP_EXTRA_INDEX} ${CMD} )
  endif()
  string( REPLACE " " ";" PYTHON_DEP_PIP_CMD "${PYTHON_DEP_PIP_CMD}" )

  # Build the full command list and convert to ----separated string
  set( PYTHON_DEP_FULL_CMD ${Python_EXECUTABLE} -m ${PYTHON_DEP_PIP_CMD} )
  string( REPLACE ";" "----" PYTHON_DEP_CMD_STR "${PYTHON_DEP_FULL_CMD}" )
  string( REPLACE ";" "----" PYTHON_DEP_ENV_STR "${PYTHON_DEP_ENV_VARS}" )

  # Use custom pip check script to avoid re-running pip on every build
  if( "${DEP}" STREQUAL "python-deps" )
    # Hash the dependency list to detect changes
    string( REPLACE ";" "," _HASH_INPUT "${VIAME_PYTHON_BASIC_DEPS}" )
    set( _HASH_FILE "${VIAME_BUILD_PREFIX}/src/python-deps-hash.txt" )

    set( PYTHON_DEP_BUILD
      ${CMAKE_COMMAND}
        -DHASH_INPUT:STRING=${_HASH_INPUT}
        -DHASH_FILE:PATH=${_HASH_FILE}
        -DPIP_INSTALL_CMD:STRING=${PYTHON_DEP_CMD_STR}
        -DENV_VARS:STRING=${PYTHON_DEP_ENV_STR}
        -P ${VIAME_CMAKE_DIR}/custom_pip_check_install.cmake
      )
  elseif( "${DEP}" STREQUAL "pytorch" OR "${DEP}" STREQUAL "torchvision" )
    # Hash the package version to detect changes (includes extra-index-url in CMD)
    set( _HASH_FILE "${VIAME_BUILD_PREFIX}/src/${DEP}-hash.txt" )

    set( PYTHON_DEP_BUILD
      ${CMAKE_COMMAND}
        -DPKG_NAME:STRING=${DEP}
        -DHASH_INPUT:STRING=${CMD}
        -DHASH_FILE:PATH=${_HASH_FILE}
        -DPIP_INSTALL_CMD:STRING=${PYTHON_DEP_CMD_STR}
        -DENV_VARS:STRING=${PYTHON_DEP_ENV_STR}
        -P ${VIAME_CMAKE_DIR}/custom_pip_check_install.cmake
      )
  else()
    set( PYTHON_DEP_BUILD
      ${CMAKE_COMMAND}
        -DCOMMAND_TO_RUN:STRING=${PYTHON_DEP_CMD_STR}
        -DENV_VARS:STRING=${PYTHON_DEP_ENV_STR}
        -P ${VIAME_CMAKE_DIR}/run_python_command.cmake
      )
  endif()

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

