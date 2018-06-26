###
# top level build file for NOAA VIAME
#

cmake_minimum_required( VERSION 3.3 )
set( CMAKE_CXX_STANDARD 11 )

project( VIAME )

include( CMakeDependentOption )


# If VIAME_FORCEBUILD is False, we will assume we dont need to rebuild any
# other superbuild proejct (except for viame). This makes development builds
# much quicker.
option( VIAME_FORCEBUILD "Enable force-building of all subpackages" ON )
mark_as_advanced( VIAME_FORCEBUILD )

###
# Add paths used throughout CMakeLists files
##
list( INSERT CMAKE_MODULE_PATH 0    "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${KWIVER_CMAKE_DIR}" )

set( VIAME_SOURCE_DIR               "${CMAKE_SOURCE_DIR}" )
set( VIAME_BINARY_DIR               "${CMAKE_BINARY_DIR}/build" )
set( VIAME_BUILD_PREFIX             "${CMAKE_BINARY_DIR}/build" )
set( VIAME_BUILD_INSTALL_PREFIX     "${CMAKE_BINARY_DIR}/install" )

set( VIAME_PACKAGES_DIR             "${VIAME_SOURCE_DIR}/packages" )
set( VIAME_DOWNLOAD_DIR             "${VIAME_PACKAGES_DIR}/downloads" )

###
# Flags relating to examples
##
option( VIAME_INSTALL_EXAMPLES      "Install existing VIAME examples"     ON )
option( VIAME_DOWNLOAD_MODELS       "Download example detection models"   ON )

if( VIAME_DOWNLOAD_MODELS )
  option( VIAME_DOWNLOAD_MODELS-HABCAM   "Download habcam detection models"   ON )
  option( VIAME_DOWNLOAD_MODELS-MOUSS    "Download mouss detection models"    ON )
  option( VIAME_DOWNLOAD_MODELS-PYTORCH  "Download pytorch models"            ON )

  mark_as_advanced( VIAME_DOWNLOAD_MODELS-HABCAM )
  mark_as_advanced( VIAME_DOWNLOAD_MODELS-MOUSS )
  mark_as_advanced( VIAME_DOWNLOAD_MODELS-PYTORCH )
else()
  set( VIAME_DOWNLOAD_MODELS-HABCAM  OFF CACHE BOOL "Forced off" FORCE )
  set( VIAME_DOWNLOAD_MODELS-MOUSS   OFF CACHE BOOL "Forced off" FORCE )
  set( VIAME_DOWNLOAD_MODELS-PYTORCH OFF CACHE BOOL "Forced off" FORCE )
endif()

###
# GPU and FFMPEG utilization flags used across projects
##
option( VIAME_ENABLE_CUDA           "Enable CUDA-Dependent Code"    OFF )
option( VIAME_ENABLE_CUDNN          "Enable CUDNN-Dependent Code"   OFF )
option( VIAME_ENABLE_FFMPEG         "Enable FFMPEG-Dependent Code"  OFF )

###
# Add core utilities enable flags
##
option( VIAME_ENABLE_DOCS           "Enable Documentation"          OFF )
option( VIAME_ENABLE_VIVIA          "Enable Vivia GUIs"             ON )
option( VIAME_ENABLE_FLASK          "Enable Flask GUIs"             OFF )
option( VIAME_ENABLE_KWANT          "Enable KW Scoring Code"        ON )

###
# Add algorithmic plugin enable flags
##
option( VIAME_ENABLE_VXL            "Enable VXL plugins"            ON )
option( VIAME_ENABLE_OPENCV         "Enable OpenCV plugins"         ON )
option( VIAME_ENABLE_MATLAB         "Enable Matlab plugins"         OFF )
option( VIAME_ENABLE_PYTHON         "Enable Python plugins"         ON )
option( VIAME_ENABLE_CAFFE          "Enable Caffe plugins"          OFF )
option( VIAME_ENABLE_PYTORCH        "Enable PyTorch plugins"        ON )
option( VIAME_ENABLE_FASTER_RCNN    "Enable Faster RCNN plugin"     OFF )
option( VIAME_ENABLE_SCALLOP_TK     "Enable ScallopTK plugin"       OFF )
option( VIAME_ENABLE_UW_PREDICTOR   "Enable UW predictor plugin"    OFF )
option( VIAME_ENABLE_YOLO           "Enable YOLO (darknet) plugin"  ON )
option( VIAME_ENABLE_BURNOUT        "Enable Burn-Out plugins"       ON )
option( VIAME_ENABLE_SMQTK          "Enable SMQTK plugins"          ON )
option( VIAME_ENABLE_CAMTRAWL       "Enable Camtrawl plugin"        OFF )


###
# Advanced hidden flags used for disabling core features
#
# These are meant for use by advanced users when you only want to use VIAME
# to build one of the above packages by itself, not VIAME core.
##
option( VIAME_ENABLE_KWIVER         "Enable KWIVER pipelining"      ON )
option( VIAME_ENABLE_VIAME_PLUGINS  "Enable VIAME plugins"          ON )

mark_as_advanced( VIAME_ENABLE_KWIVER )
mark_as_advanced( VIAME_ENABLE_VIAME_PLUGINS )

###
# Add external package build instructions
##
option( VIAME_BUILD_DEPENDENCIES    "Build all required dependencies in a super-build" ON )
option( VIAME_CREATE_PACKAGE        "Build a binary package at the end of build"       OFF )
mark_as_advanced( VIAME_CREATE_PACKAGE )

###
# Add extra options
##
option( VIAME_BUILD_CORE_IMAGE_LIBS "Build core image libraries such as libtiff and libjpeg" ON )
mark_as_advanced( VIAME_BUILD_CORE_IMAGE_LIBS )

option( VIAME_BUILD_TESTS "Build VIAME and all sub-packages with tests enabled" OFF )
mark_as_advanced( VIAME_BUILD_TESTS )

###
# Add logic relating to enable flags
##
if( VIAME_ENABLE_CAFFE OR VIAME_ENABLE_SCALLOP_TK OR VIAME_ENABLE_BURNOUT OR
    VIAME_ENABLE_FASTER_RCNN OR VIAME_ENABLE_YOLO )
  set( VIAME_ENABLE_OPENCV  ON CACHE BOOL "OpenCV required for other projects"   FORCE )
endif()

if( VIAME_ENABLE_FASTER_RCNN OR VIAME_ENABLE_SMQTK OR VIAME_ENABLE_PYTORCH )
  set( VIAME_ENABLE_PYTHON  ON CACHE BOOL "Python required for other projects"   FORCE )
endif()

if( VIAME_ENABLE_FASTER_RCNN )
  set( VIAME_ENABLE_CAFFE   ON CACHE BOOL "Caffe required for other projects"    FORCE )
endif()

if( VIAME_ENABLE_KWANT OR VIAME_ENABLE_BURNOUT OR VIAME_ENABLE_VIVIA )
  set( VIAME_ENABLE_VXL     ON CACHE BOOL "VXL required for other projects"      FORCE )
endif()

if( VIAME_ENABLE_VIVIA )
  set( VIAME_ENABLE_BURNOUT ON CACHE BOOL "Burn-Out required for other projects" FORCE )
endif()

if( VIAME_ENABLE_MATLAB )
  find_package( Matlab REQUIRED COMPONENTS ENG_LIBRARY MX_LIBRARY )
endif()

if( VIAME_ENABLE_DOCS )
  find_package( Doxygen REQUIRED )
endif()

if( VIAME_ENABLE_PYTHON )
  option( VIAME_DISABLE_PYTHON_CHECK   "Disable python version checking" OFF )
  mark_as_advanced( VIAME_DISABLE_PYTHON_CHECK )

  if (NOT WIN32)
    option( VIAME_SYMLINK_PYTHON "Symlink python files instead of copying." OFF )
    mark_as_advanced( VIAME_SYMLINK_PYTHON )
  endif()
  set(KWIVER_SYMLINK_PYTHON ${VIAME_SYMLINK_PYTHON})

  find_package( PythonInterp REQUIRED )
  find_package( PythonLibs REQUIRED )

  if( NOT ${VIAME_DISABLE_PYTHON_CHECK} AND
      NOT EXISTS ${PYTHON_INCLUDE_DIR}/pyconfig.h )
    message( FATAL_ERROR "pyconfig.h does not exist in your PYTHON_INCLUDE_DIR \
      make sure the variable is set correctly and you have python-dev installed." )
  endif()

  set( PYTHON_VERSION "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
       CACHE STRING "Python version number" )
endif()

# VXL enabled triggers yasm to be built, requiring at least some python
if( VIAME_ENABLE_VXL )
  find_package( PythonInterp QUIET )
  if( NOT PYTHONINTERP_FOUND )
    message( FATAL_ERROR "If VXL is enabled, there must be some python install located \
      on the system to build, even if VIAME_ENABLE_PYTHON is turned off for all other \
      packages. A python binary could not be found. Please install at least python." )
  endif()
endif()

if( VIAME_ENABLE_CUDA )
  find_package( CUDA QUIET REQUIRED )

  option(VIAME_FORCE_CUDA_CSTD98 "Build all cuda code with C++ standard 98" FALSE)
  mark_as_advanced(VIAME_FORCE_CUDA_CSTD98)
else()
  set(VIAME_FORCE_CUDA_CSTD98 CACHE INTERNAL FALSE)
endif()

if( VIAME_ENABLE_CUDNN AND NOT VIAME_ENABLE_CUDA )
  message( FATAL_ERROR "Cannot enable CUDNN without CUDA support" )
endif()

if( WIN32 AND VIAME_ENABLE_FASTER_RCNN )
  message( FATAL_ERROR "Faster R-CNN on windows is currently untested \
    in VIAME, either disable it, or if you're ambitious, disable this message \
    and continue." )
endif()

if( VIAME_ENABLE_CUDNN )
  set( CUDNN_ROOT "" CACHE PATH "CUDNN root folder, leave as blank to auto-detect." )

  find_library( CUDNN_LIBRARY REQUIRED
    NAMES cudnn libcudnn.so libcudnn.so.5
    PATHS ${CUDNN_ROOT}
          ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux
          ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 )

  if( NOT CUDNN_LIBRARY )
    message( FATAL_ERROR "Unable to locate CUDNN library" )
  else()
    # Different subprojects use each variable
    set( CUDNN_LIBRARIES CUDNN_LIBRARY CACHE INTERNAL "" FORCE )
  endif()
endif()

if( WIN32 AND VIAME_ENABLE_PYTHON AND CMAKE_CONFIGURATION_TYPES EQUAL "Debug" )
  message( FATAL_ERROR "Cannot build in Debug on Windows with Python enabled, \
    build in RelWithDebInfo or Release until fixed." )
endif()

if( VIAME_ENABLE_OPENCV )
  set( VIAME_OPENCV_VERSION 2.4.13 CACHE STRING "Version of OpenCV to use" )
  set_property( CACHE VIAME_OPENCV_VERSION PROPERTY STRINGS 2.4.13 3.1.0 3.3.0 )

  # Version 2.4.11 no longer present fletch
  if( ${VIAME_OPENCV_VERSION} EQUAL "2.4.11" )
    set( VIAME_OPENCV_VERSION "2.4.13" CACHE STRING "Version of OpenCV to use" FORCE )
  endif()
endif()

# Set default windows configuration mode to be 'Release'
set( VIAME_FIRST_CONFIGURATION TRUE CACHE BOOL "Has this project been configured." )
mark_as_advanced( VIAME_FIRST_CONFIGURATION )

if( MSVC AND VIAME_FIRST_CONFIGURATION )
  set( CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "This specifies what \
    build types (configurations) will be available for selection within \
	Visual Studio. Options are: Release, RelWithDebInfo, Debug etc. in a \
	semi-colon seperated list." FORCE )

  set( VIAME_FIRST_CONFIGURATION FALSE CACHE BOOL "Project has been configured." FORCE )
endif()

if( MSVC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "18.00" )
  message( FATAL_ERROR "You must use at least MSVC version 13 or greater" )
endif()
# if we have a 32 bit build, turn off Eigen alignment for more information see:
# http://eigen.tuxfamily.org/dox/group__TopicUnalignedArrayAssert.html
if( CMAKE_SIZEOF_VOID_P EQUAL 4 )
  set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_VECTORIZE -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT")
endif()

# Include internal common CMake macros used across all CMakeLists.txt
include( common_macros )

# Include checks across all prior versions of the code
include( old_version_checks )

###
# Main build loop for when we are building both VIAME and all dependencies
##
if( VIAME_BUILD_DEPENDENCIES )

  # Early error checking in case user did not run submodule init
  file( GLOB fletch_files ${VIAME_PACKAGES_DIR}/fletch/* )
  list( LENGTH fletch_files fletch_length )

  if( ${fletch_length} LESS 3 )
    message( FATAL_ERROR "Fletch directory empty, did you not run the following in your source tree: \n\
     git submodule update --init --recursive" )
  endif()

  # Include for CMake external project add command
  include( ExternalProject )

  # Specifies internal arguments passed to all projects (CXX flag passdowns, etc)
  include( common_args )

  # Hard VIAME requirements
  include( add_project_fletch )

  # Add optional projects
  if( VIAME_ENABLE_YOLO )
    include( add_project_darknet )
  endif()

  if( VIAME_ENABLE_BURNOUT )
    include( add_project_burnout )
  endif()

  # Add core (required) kwiver requirement
  if( VIAME_ENABLE_KWIVER )
    include( add_project_kwiver )
  endif()

  # Add other optional libraries
  if( VIAME_ENABLE_SCALLOP_TK )
    include( add_project_scallop_tk )
  endif()

  if( VIAME_ENABLE_VIVIA )
    include( add_project_vivia )
  endif()

  if( VIAME_ENABLE_KWANT )
    include( add_project_kwant )
  endif()

  if( VIAME_ENABLE_FASTER_RCNN )
    include( add_project_faster_rcnn )
  endif()

  if( VIAME_ENABLE_SMQTK )
    include( add_project_smqtk )
  endif()

  if( VIAME_ENABLE_PYTORCH )
    include( add_project_pytorch )
  endif()

  # VIAME plugins, adds this project causing the below else() block to run
  if( VIAME_ENABLE_VIAME_PLUGINS )
    include( add_project_viame )
  endif()

  # Fixup installer package
  if( VIAME_CREATE_PACKAGE )
    include( build_package )
  endif()
else()

if( VIAME_ENABLE_FLASK )
  set(FT_DATA_ROOT "${CMAKE_BINARY_DIR}/data" CACHE STRING "Where fishtracker data is stored")
  file(COPY "plugins/flask"
       DESTINATION "plugins/flask"
       PATTERN "*.in" EXCLUDE)
  configure_file("plugins/flask/flask/fishtracker.py.in"
                 "plugins/flask/flask/fishtracker.py"
                 )
  configure_file("plugins/flask/Python/run_python_script.sh.in"
                 "plugins/flask/Python/run_python_script.sh"
                 )
endif()

  ###
  # Look for and use Fletch to find dependencies
  ##
  find_package( kwiver REQUIRED )
  include_directories( ${KWIVER_INCLUDE_DIRS} )
  link_directories( ${KWIVER_LIBRARY_DIR} )
  list( INSERT CMAKE_MODULE_PATH  0         "${KWIVER_CMAKE_DIR}" )

  if( IS_DIRECTORY ${fletch_DIR} )
    find_package( fletch NO_MODULE )
    list( APPEND CMAKE_PREFIX_PATH  "${fletch_DIR}" )
  endif()

  if( VIAME_ENABLE_OPENCV )
    find_package( OpenCV REQUIRED )
  endif()

  include_directories( "${CMAKE_CURRENT_BINARY_DIR}" "${KWIVER_INCLUDE_DIRS}" )

  ###
  # KWIVER CMake config
  ##
  include( kwiver-cmake-future )
  include( kwiver-utils )

  ##
  # System specific compiler flags
  include( viame-flags )

  ##
  # Build system hack for visual system, will be removed
  include( windows-ksys-hack )

  ##
  # check compiler support
  include( kwiver-configcheck )

  # set the name for our package exports
  set( kwiver_export_name      viame_exports )

  # Setup python directories for plugin processes in same style as KWIVER
  if( VIAME_ENABLE_PYTHON )
    set( kwiver_python_subdir "python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}" )
    set( kwiver_python_output_path "${VIAME_BUILD_PREFIX}/lib/${kwiver_python_subdir}" )

    kwiver_create_python_plugin_init( viame )
    kwiver_create_python_plugin_init( viame/processes )
  endif()

  ###
  # build submodules

  ###
  # add subdirs
  ##
  add_subdirectory( plugins )
  add_subdirectory( examples )
  add_subdirectory( configs )

  ###
  # Configure setup scripts
  ##
  include( setup_viame_paths )

  if( WIN32 )
    set( VIAME_PIPELINE_RUNNER_SCRIPT   "${VIAME_BINARY_DIR}/pipeline_runner.bat" )
    set( VIAME_PROCESSOPEDIA_SCRIPT     "${VIAME_BINARY_DIR}/processopedia.bat" )
    set( VIAME_PLUGIN_EXPLORER_SCRIPT   "${VIAME_BINARY_DIR}/plugin_explorer.bat" )
    set( VIAME_SETUP_SCRIPT             "${VIAME_BINARY_DIR}/setup_viame.bat" )

    configure_file(
      ${CMAKE_SOURCE_DIR}/cmake/pipeline_runner.bat.in
      ${VIAME_PIPELINE_RUNNER_SCRIPT}
    )
    configure_file(
      ${CMAKE_SOURCE_DIR}/cmake/processopedia.bat.in
      ${VIAME_PROCESSOPEDIA_SCRIPT}
    )
    configure_file(
      ${CMAKE_SOURCE_DIR}/cmake/plugin_explorer.bat.in
      ${VIAME_PLUGIN_EXPLORER_SCRIPT}
    )
    configure_file(
      ${CMAKE_SOURCE_DIR}/cmake/setup_viame.bat.in
      ${VIAME_SETUP_SCRIPT}
    )

    install( PROGRAMS      ${VIAME_PIPELINE_RUNNER_SCRIPT}
                           ${VIAME_PROCESSOPEDIA_SCRIPT}
                           ${VIAME_PLUGIN_EXPLORER_SCRIPT}
                           ${VIAME_SETUP_SCRIPT}
             DESTINATION   ${CMAKE_INSTALL_PREFIX} )
  else()
    set( VIAME_SETUP_SCRIPT_FILE    "${VIAME_BINARY_DIR}/setup_viame.sh" )

    configure_file(
      ${CMAKE_SOURCE_DIR}/cmake/setup_viame.sh.in
      ${VIAME_SETUP_SCRIPT_FILE}
    )

    install( PROGRAMS      ${VIAME_SETUP_SCRIPT_FILE}
             DESTINATION   ${CMAKE_INSTALL_PREFIX} )
  endif()

  ###
  # Install system libs if packaging enabled
  ##
  if( VIAME_CREATE_PACKAGE )
    set( CMAKE_INSTALL_UCRT_LIBRARIES TRUE )
    include( InstallRequiredSystemLibraries )
    if( CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS )
      install( PROGRAMS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS} DESTINATION bin COMPONENT System )
    endif()
  endif()
endif()
