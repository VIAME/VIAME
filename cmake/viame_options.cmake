###
# VIAME's options
#
# Every `option()` VIAME declares, and the dependent-option logic that goes
# with them: a flag that forces another on, a flag that is meaningless on a
# platform, a flag whose default depends on another.
#
# Lifted out of the top-level `CMakeLists.txt` by P1-T02, unchanged.
# `lite-build-system.md` section 2 is the list this file is converging on,
# and says which options go and in which phase.
##

###
# GPU utilization flags used across projects
##
option( VIAME_ENABLE_CUDA           "Enable CUDA-Dependent Code"    ON )
option( VIAME_ENABLE_CUDNN          "Enable CUDNN-Dependent Code"   ON )

###
# Add core user interface enable flags
##
option( VIAME_ENABLE_DIVE           "Enable DIVE GUI"               ON )

# `VIAME_ENABLE_VIVIA`, `VIAME_ENABLE_SEAL` and `VIAME_ENABLE_KEYPOINT` stood
# here. Each named a separate repository the superbuild cloned and built
# beside VIAME; `lite-build-system.md` section 2 removes all three in P1.
# VIVIA's needed VXL, which went in P3, and had already been a fatal error
# since then.

###
# Add default-enabled algorithmic plugin enable flags
##
option( VIAME_ENABLE_PYTHON         "Enable Python plugins"         ON )
option( VIAME_ENABLE_PYTORCH        "Enable PyTorch plugins"        ON )
option( VIAME_ENABLE_ONNX           "Enable ONNX runtime plugins"   ON )
option( VIAME_ENABLE_OPENCV         "Enable OpenCV plugins"         ON )
option( VIAME_ENABLE_DARKNET        "Enable Darknet (YOLO) plugin"  ON )
option( VIAME_ENABLE_SVM            "Enable SVM plugins"            ON )

###
# Add secondary algorithm plugin enable flags (non-advanced)
##
# `VIAME_ENABLE_TENSORFLOW` and `VIAME_ENABLE_TENSORRT` stood here, removed
# in P1 with the rest of section 2's list: two inference backends that no
# shipped pipeline selects, each pulling a runtime of its own.
#
# `VIAME_ENABLE_MATLAB` stays for now. It is on the same list, but it is also
# **open decision 4** in `lite-plan.md`, which proposes the drop rather than
# taking it, and `plugins/matlab` is two working bridges (camtrawl,
# annosaurus) rather than a build-system artefact. The option and the code
# go when that decision is taken.
option( VIAME_ENABLE_MATLAB         "Enable Matlab plugins"         OFF )

###
# Add tertiary plugin enable flags (advanced)
##
# `VIAME_ENABLE_GDAL` stood here. It only ever told fletch to build GDAL, for
# the VXL image reader that went in P3; nothing in VIAME has read it since.
option( VIAME_ENABLE_SEAGIS         "Enable SEAGIS StereoLib"       OFF )
option( VIAME_ENABLE_COLMAP         "Enable COLMAP regisration"     OFF )

mark_as_advanced( VIAME_ENABLE_SEAGIS )
mark_as_advanced( VIAME_ENABLE_COLMAP )

###
# Add core utilities enable flags
##
option( VIAME_ENABLE_DOCS           "Enable Documentation"          OFF )

# `VIAME_ENABLE_WIN32GUI` stood here, and was read in one place: whether
# fletch built Qt. It went with the GUIs.

###
# Flags relating to examples and model downloads
##
option( VIAME_INSTALL_EXAMPLES      "Install existing VIAME examples"     ON )
option( VIAME_DOWNLOAD_MODELS       "Download example detection models"   ON )

####
# This section generates misc VIAME_DOWNLOAD_MODEL-[NAME] advanced flags
##
if( VIAME_DOWNLOAD_MODELS )
  # Parse addon model pack options from CSV file
  # Options are declared here but downloads/installs happen in configs/add-ons
  ParseModelDownloadOptions( ${VIAME_CMAKE_DIR}/download_viame_addons.csv )

  # Auto-detect addon folders not in the CSV and declare options for them
  ModelDownloadOptionsFromFolders( "${CMAKE_SOURCE_DIR}/configs/add-ons" )

  # Standalone seed weights, not backed by an addon folder or model pack
  option( VIAME_DOWNLOAD_MODELS-EFFICIENTNETV2M
    "EfficientNetV2-M classifier training seed model" OFF )
  mark_as_advanced( VIAME_DOWNLOAD_MODELS-EFFICIENTNETV2M )
  option( VIAME_DOWNLOAD_MODELS-EFFICIENTNETV2L
    "EfficientNetV2-L classifier training seed model" OFF )
  mark_as_advanced( VIAME_DOWNLOAD_MODELS-EFFICIENTNETV2L )
else()
  # Force all flags to OFF in the event that the root flag is set
  DisableAllModelDownloads()
endif()

###
# Additional libraries built on pytorch and versioning
##
if( VIAME_ENABLE_PYTORCH )
  set( VIAME_PYTORCH_VERSION 2.12.0 CACHE STRING "PyTorch version to use" )
  set_property( CACHE VIAME_PYTORCH_VERSION PROPERTY STRINGS "1.13.1" "2.12.0" )
  mark_as_advanced( VIAME_PYTORCH_VERSION )

  set( PYTORCH_INTERNAL_VERSION 2.12.0 CACHE INTERNAL "Internal pytorch version" )
  set( PYTORCH_MIN_GCC          11.3  CACHE INTERNAL "Minimum GCC version for torch" )
  set( PYTORCH_MIN_PYTHON_WHL   3.10  CACHE INTERNAL "Minimum python for torch whl" )
  set( PYTORCH_MIN_PYTHON_BLD   3.10  CACHE INTERNAL "Minimum python for torch build" )
  set( PYTORCH_MIN_CUDA_BLD     11.0  CACHE INTERNAL "Minimum cuda for torch build" )
  set( PYTORCH_MIN_CUDNN_BLD    7.0   CACHE INTERNAL "Minimum cudnn for torch build" )

  option( VIAME_ENABLE_PYTORCH-VISION      "Enable TorchVision algorithms"  ON )
  option( VIAME_ENABLE_PYTORCH-MMDET       "Enable mmdet algorithms"        ON )
  option( VIAME_ENABLE_PYTORCH-NETHARN     "Enable netharn algorithms"      ON )
  option( VIAME_ENABLE_PYTORCH-MIT-YOLO    "Enable mit-yolo PyTorch code"   ON )
  option( VIAME_ENABLE_PYTORCH-HUGGINGFACE "Enable HuggingFace algorithm"   ON )
  option( VIAME_ENABLE_PYTORCH-RF-DETR     "Enable RF-DETR algorithms"      ON )
  option( VIAME_ENABLE_PYTORCH-LEARN       "Enable LEARN plugins"           OFF )
  option( VIAME_ENABLE_PYTORCH-VIDEO       "Enable video algorithms"        OFF )
  option( VIAME_ENABLE_PYTORCH-SIAMMASK    "Enable siammask algorithms"     OFF )
  option( VIAME_ENABLE_PYTORCH-DETECTRON2  "Enable detectron2 algorithms"   OFF )
  option( VIAME_ENABLE_PYTORCH-MDNET       "Enable mdnet algorithms"        OFF )
  option( VIAME_ENABLE_PYTORCH-SAM2        "Enable SAM2 algorithms"         OFF )
  option( VIAME_ENABLE_PYTORCH-SAM3        "Enable SAM3 algorithms"         OFF )
  option( VIAME_ENABLE_PYTORCH-STEREO      "Enable stereo algorithms"       OFF )
  option( VIAME_ENABLE_PYTORCH-ULTRALYTICS "Enable ultralytics algorithms"  OFF )
  option( VIAME_ENABLE_PYTORCH-LITDET      "Enable LitDet algorithms"       OFF )
  option( VIAME_ENABLE_PYTORCH-DINO3       "Enable DINOv3 algorithms"       OFF )

  mark_as_advanced( VIAME_ENABLE_PYTORCH-VISION )
  mark_as_advanced( VIAME_ENABLE_PYTORCH-MMDET )
  mark_as_advanced( VIAME_ENABLE_PYTORCH-NETHARN )
  mark_as_advanced( VIAME_ENABLE_PYTORCH-MIT-YOLO )
  mark_as_advanced( VIAME_ENABLE_PYTORCH-HUGGINGFACE )
  mark_as_advanced( VIAME_ENABLE_PYTORCH-LEARN )
  mark_as_advanced( VIAME_ENABLE_PYTORCH-VIDEO )
  mark_as_advanced( VIAME_ENABLE_PYTORCH-SIAMMASK )
  mark_as_advanced( VIAME_ENABLE_PYTORCH-DETECTRON2 )
  mark_as_advanced( VIAME_ENABLE_PYTORCH-MDNET )
  mark_as_advanced( VIAME_ENABLE_PYTORCH-SAM2 )
  mark_as_advanced( VIAME_ENABLE_PYTORCH-SAM3 )
  mark_as_advanced( VIAME_ENABLE_PYTORCH-STEREO )
  mark_as_advanced( VIAME_ENABLE_PYTORCH-ULTRALYTICS )
  mark_as_advanced( VIAME_ENABLE_PYTORCH-RF-DETR )
  mark_as_advanced( VIAME_ENABLE_PYTORCH-LITDET )
  mark_as_advanced( VIAME_ENABLE_PYTORCH-DINO3 )

  option( VIAME_BUILD_PYTORCH_FROM_SOURCE      "Build PyTorch from source"     OFF )
  option( VIAME_BUILD_TORCHVISION_FROM_SOURCE  "Build TorchVision from source" ON )

  mark_as_advanced( VIAME_BUILD_PYTORCH_FROM_SOURCE )
  mark_as_advanced( VIAME_BUILD_TORCHVISION_FROM_SOURCE )

  if( NOT CMAKE_GENERATOR STREQUAL "Ninja" )
    option( VIAME_BUILD_LIMIT_NINJA    "Limit ninja use on sensitive projects" ON )
    mark_as_advanced( VIAME_BUILD_LIMIT_NINJA )
  endif()

  if( NOT VIAME_ENABLE_PYTORCH )
    set( VIAME_BUILD_PYTORCH_FROM_SOURCE OFF CACHE BOOL "Forced off" FORCE )
  endif()
  if( NOT VIAME_ENABLE_PYTORCH-VISION )
    set( VIAME_BUILD_TORCHVISION_FROM_SOURCE  OFF CACHE BOOL "Forced off" FORCE )
  endif()
endif()

###
# Core build settings advanced flags
##

# Alternative build directories for specific components
set( VIAME_BUILD_FLETCH_DIR "${VIAME_BUILD_PREFIX}/src/fletch-build"
     CACHE STRING "VIAME superbuild FLETCH build location" )
mark_as_advanced( VIAME_BUILD_FLETCH_DIR )

set( VIAME_BUILD_KWIVER_DIR "${VIAME_BUILD_PREFIX}/src/kwiver-build"
     CACHE STRING "VIAME superbuild KWIVER build location" )
mark_as_advanced( VIAME_BUILD_KWIVER_DIR )

set( VIAME_BUILD_PLUGINS_DIR "${VIAME_BUILD_PREFIX}/src/viame-build"
     CACHE STRING "VIAME superbuild plugins build location" )
mark_as_advanced( VIAME_BUILD_PLUGINS_DIR )

# If VIAME_BUILD_FORCE_REBUILD is False, we will rely on CMake's default system
# for testing when projects need to be rebuilt, this can make builds be (slightly)
# faster. When True, we rely on each individual projects checking for if certain
# parts of it should be rebuilt, and cmake stamps are ignored.
option( VIAME_BUILD_FORCE_REBUILD "Enable force-building of all subpackages" OFF )
mark_as_advanced( VIAME_BUILD_FORCE_REBUILD )

# Control the number of parallel build threads for subprojects like PyTorch.
# Set this to limit the number of concurrent compilation jobs (e.g., 4 or 8).
set( VIAME_BUILD_MAX_THREADS "" CACHE STRING "Parallel threads for subprojects (empty = auto)" )
mark_as_advanced( VIAME_BUILD_MAX_THREADS )

# Disable pip cache to save disk space during builds (useful for CI environments)
option( VIAME_BUILD_NO_CACHE_DIR "Disable pip cache to save disk space" OFF )
mark_as_advanced( VIAME_BUILD_NO_CACHE_DIR )

###
# Other advanced hidden flags used for disabling core features
##
# `VIAME_ENABLE_KWIVER` and `VIAME_ENABLE_VIAME_PLUGINS` stood here, for
# building one of the superbuild's packages without VIAME itself. There are
# no other packages now -- kwiver is `library/`, and the plugins are the
# project -- so both were "build nothing", and one was already a fatal error
# without the other.

# A continuation build allows building of new plugins using an existing
# VIAME build in another folder or build tree, for the purpose of making
# chained windows MSI installers
option( VIAME_BUILD_PACKAGING_CONT  "Enable continuation build"     OFF )
mark_as_advanced( VIAME_BUILD_PACKAGING_CONT )

if( VIAME_BUILD_PACKAGING_CONT )
  set( VIAME_PRIOR_BUILD "" CACHE PATH "Location of prior VIAME build" )
  mark_as_advanced( VIAME_PRIOR_BUILD )
endif()

###
# Add macro-level package build options
##
option( VIAME_BUILD_DEPENDENCIES "Build all required dependencies in a super-build" ON )
mark_as_advanced( VIAME_BUILD_DEPENDENCIES )

option( VIAME_FIXUP_BUNDLE       "Run fixup bundle on top of generated binaries"    OFF )
mark_as_advanced( VIAME_FIXUP_BUNDLE )

option( VIAME_CREATE_INSTALLER   "Build a msi prototype installer using wix"        OFF )
mark_as_advanced( VIAME_CREATE_INSTALLER )

option( VIAME_VERSION_RELEASE    "Install release scripts instead of development"   OFF )
mark_as_advanced( VIAME_VERSION_RELEASE )

if( VIAME_CREATE_INSTALLER )
  set( VIAME_FIXUP_BUNDLE ON CACHE BOOL "Flag forced due to create_package enable"  FORCE )
endif()

if( VIAME_FIXUP_BUNDLE )
  set( VIAME_VERSION_RELEASE ON CACHE BOOL "Flag forced due to fixup_bundle enable" FORCE )
endif()

###
# Add extra options
##
option( VIAME_BUILD_CORE_IMAGE_LIBS "Build core image libraries such as libpng"     ON )
mark_as_advanced( VIAME_BUILD_CORE_IMAGE_LIBS )

option( VIAME_ENABLE_TESTS "Build VIAME tests"                                      OFF )
mark_as_advanced( VIAME_ENABLE_TESTS )

option( VIAME_BUILD_CHECKS   "Enable version checks on gcc and python"              ON )
mark_as_advanced( VIAME_BUILD_CHECKS )

###
# Add logic and error checking relating to enable flags
##
if( VIAME_ENABLE_DARKNET )
  set( VIAME_ENABLE_OPENCV  ON CACHE BOOL "OpenCV required for other projects"  FORCE )
endif()

if( VIAME_ENABLE_PYTORCH )
  set( VIAME_ENABLE_PYTHON  ON CACHE BOOL "Python required for other projects"  FORCE )
endif()

if( VIAME_ENABLE_PYTORCH-LEARN )
  set( VIAME_ENABLE_PYTORCH-DETECTRON2  ON CACHE BOOL "Detectron2 required for project" FORCE )
  set( VIAME_ENABLE_PYTORCH-VIDEO      ON CACHE BOOL "Torch Video required for project" FORCE )
endif()

if( VIAME_ENABLE_MATLAB )
  find_package( Matlab REQUIRED COMPONENTS ENG_LIBRARY MX_LIBRARY )
endif()

if( VIAME_ENABLE_DOCS )
  find_package( Doxygen REQUIRED )
endif()

if( VIAME_ENABLE_PYTORCH-LEARN AND NOT VIAME_ENABLE_CUDA )
  message( FATAL_ERROR "CUDA required for LEARN project currently" )
endif()

if( WIN32 AND VIAME_BUILD_DEPENDENCIES AND VIAME_BUILD_CHECKS )
  string( LENGTH "${VIAME_BUILD_KWIVER_DIR}" KWIVER_BUILD_DIR_LENGTH )
  string( LENGTH "${VIAME_BUILD_PLUGINS_DIR}" PLUGINS_BUILD_DIR_LENGTH )

  if( KWIVER_BUILD_DIR_LENGTH GREATER 12 OR PLUGINS_BUILD_DIR_LENGTH GREATER 12 )
    message( FATAL_ERROR "VIAME_BUILD_KWIVER_DIR and VIAME_BUILD_PLUGINS_DIR \
      must be set to a short path (e.g. C:\\tmp\\kv1 and C:\\tmp\\vm1) on \
      Windows due to a current issue with nesting KWIVER exceeding the 260 \
      character default filepath size. Alternatively disable VIAME_BUILD_CHECKS \
      to ignore this message if you either increased the windows default path \
      length, or think your build path is short enough." )
  endif()
endif()
