###
# top level build file for NOAA VIAME
#

cmake_minimum_required( VERSION 3.3 )

project( VIAME )

include( CMakeDependentOption )

###
# Add paths used throughout CMakeLists files
##
list( INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/CMake" "${KWIVER_CMAKE_DIR}" )

set( VIAME_SOURCE_DIR            "${CMAKE_SOURCE_DIR}" )
set( VIAME_BINARY_DIR            "${CMAKE_BINARY_DIR}/build" )
set( VIAME_BUILD_PREFIX          "${CMAKE_BINARY_DIR}/build" )
set( VIAME_BUILD_INSTALL_PREFIX  "${CMAKE_BINARY_DIR}/install" )

set( VIAME_PACKAGES_DIR          "${VIAME_SOURCE_DIR}/packages" )
set( VIAME_DOWNLOAD_DIR          "${VIAME_PACKAGES_DIR}/downloads" )

###
# Flags relating to examples
##
option( VIAME_INSTALL_EXAMPLES    "Install existing VIAME examples"     ON )
option( VIAME_DOWNLOAD_MODELS     "Download example detection models"   ON )

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
option( VIAME_ENABLE_VIVIA          "Enable Vivia GUIs"             OFF )
option( VIAME_ENABLE_KWANT          "Enable KW Scoring Code"        OFF )

###
# Add algorithmic plugin enable flags
##
option( VIAME_ENABLE_VXL            "Enable VXL plugins"            OFF )
option( VIAME_ENABLE_OPENCV         "Enable OpenCV plugins"         OFF )
option( VIAME_ENABLE_MATLAB         "Enable Matlab plugins"         OFF )
option( VIAME_ENABLE_PYTHON         "Enable Python plugins"         OFF )
option( VIAME_ENABLE_CAFFE          "Enable Caffe plugins"          OFF )
option( VIAME_ENABLE_FASTER_RCNN    "Enable Faster RCNN plugin"     OFF )
option( VIAME_ENABLE_SCALLOP_TK     "Enable ScallopTK plugin"       OFF )
option( VIAME_ENABLE_UW_PREDICTOR   "Enable UW predictor plugin"    OFF )
option( VIAME_ENABLE_YOLO           "Enable YOLO (darknet) plugin"  OFF )
option( VIAME_ENABLE_BURNOUT        "Enable Burn-Out plugin"        OFF )

###
# Add external package build instructions
##
option( VIAME_BUILD_DEPENDENCIES  "Build all required dependencies in a super-build" ON )

###
# Add logic relating to enable flags
##
if( VIAME_ENABLE_CAFFE OR VIAME_ENABLE_SCALLOP_TK OR VIAME_ENABLE_MATLAB OR
    VIAME_ENABLE_FASTER_RCNN )
  set( VIAME_ENABLE_OPENCV ON CACHE BOOL "OpenCV required for other projects" FORCE )
endif()

if( VIAME_ENABLE_FASTER_RCNN )
  set( VIAME_ENABLE_PYTHON ON CACHE BOOL "Python required for other projects" FORCE )
endif()

if( VIAME_ENABLE_FASTER_RCNN )
  set( VIAME_ENABLE_CAFFE ON CACHE BOOL "Caffe required for other projects" FORCE )
endif()

if( VIAME_ENABLE_KWANT OR VIAME_ENABLE_VIVIA )
  set( VIAME_ENABLE_VXL ON CACHE BOOL "VXL required for other projects" FORCE )
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
  option( VIAME_DISABLE_PYTHON_CHECK   "Disable python version checking" OFF)
  mark_as_advanced( VIAME_DISABLE_PYTHON_CHECK )

  find_package( PythonLibs REQUIRED )
  find_package( PythonInterp REQUIRED )

  if( NOT ${VIAME_DISABLE_PYTHON_CHECK} AND
      ( NOT PYTHON_VERSION_MAJOR EQUAL "2" OR NOT PYTHON_VERSION_MINOR EQUAL "7" ) )
    message( FATAL_ERROR "Only Python2.7.* is currently supported, please point  \
      your PYTHON_EXECUTABLE, PYTHON_LIBRARY and PYTHON_INCLUDE_DIR to a  \
      Python2.7.* install, or set VIAME_DISABLE_PYTHON_CHECK to ignore this \
      message and build with other versions at your own peril." )
  endif()

  if( NOT ${VIAME_DISABLE_PYTHON_CHECK} AND
      NOT PYTHONLIBS_VERSION_STRING VERSION_LESS "3.0.0" )
    message( FATAL_ERROR "It appears your pipeline executable and headers point \
      to different versions of Python. Only Python2.7.* is currently supported, \
      please point your PYTHON_LIBRARY and PYTHON_INCLUDE_DIR to a Python2.7.* \
      install, or set VIAME_DISABLE_PYTHON_CHECK to ignore this message." )
  endif()

  if( NOT ${VIAME_DISABLE_PYTHON_CHECK} AND
      NOT EXISTS ${PYTHON_INCLUDE_DIR}/pyconfig.h )
    message( FATAL_ERROR "pyconfig.h does not exist in your PYTHON_INCLUDE_DIR \
      make sure the variable is set correctly and you have python-dev installed." )
  endif()
endif()

if( VIAME_ENABLE_CUDA )
  find_package( CUDA QUIET REQUIRED )
endif()

if( VIAME_ENABLE_CUDNN AND NOT VIAME_ENABLE_CUDA )
  message( FATAL_ERROR "Cannot enable CUDNN without CUDA support" )
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

if( WIN32 AND VIAME_ENABLE_CAFFE )
  message( FATAL_ERROR "Internal Caffe build on windows not currently supported, \
    please build Caffe externally using the Caffe windows branch, if desired." )
endif()

if( WIN32 AND VIAME_ENABLE_PYTHON AND CMAKE_CONFIGURATION_TYPES EQUAL "Debug" )
  message( FATAL_ERROR "Cannot build in Debug on Windows with Python enabled, \
    build in RelWithDebInfo or Release until fixed." )
endif()

if( VIAME_ENABLE_OPENCV )
  set( VIAME_OPENCV_VERSION 2.4.11 CACHE STRING "Version of OpenCV to use" )
  set_property( CACHE VIAME_OPENCV_VERSION PROPERTY STRINGS 2.4.11 3.1.0 )
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

# Include internal common CMake macros used across all CMakeLists.txt
include( common_macros )

###
# Main build loop for when we are building both VIAME and all dependencies
##
if( VIAME_BUILD_DEPENDENCIES )

  # Include for CMake external project add command
  include( ExternalProject )

  # Specifies internal arguments passed to all projects (CXX flag passdowns, etc)
  include( common_args )

  # Hard VIAME requirements
  include( add_project_fletch )

  # Add optional YOLO build
  if( VIAME_ENABLE_YOLO )
    include( add_project_darknet )
  endif()

  # Add core (required) kwiver requirement
  include( add_project_kwiver )

  # Add other optional libraries
  if( VIAME_ENABLE_SCALLOP_TK )
    include( add_project_scallop_tk )
  endif()

  if( VIAME_ENABLE_BURNOUT )
    include( add_project_burnout )
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

  # VIAME plugins, adds this project causing the below else() block to run
  include( add_project_viame )

else()

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

  ###
  # build submodules

  ###
  # add subdirs
  ##
  add_subdirectory( plugins )
  add_subdirectory( examples )

  ###
  # Configure setup scripts
  ##
  include( setup_viame_paths )

  if( WIN32 )
    set( VIAME_PIPELINE_RUNNER_SCRIPT   "${VIAME_BINARY_DIR}/pipeline_runner.bat" )
    set( VIAME_PROCESSOPEDIA_SCRIPT     "${VIAME_BINARY_DIR}/processopedia.bat" )
    set( VIAME_PLUGIN_EXPLORER_SCRIPT   "${VIAME_BINARY_DIR}/plugin_explorer.bat" )

    configure_file(
      ${CMAKE_SOURCE_DIR}/CMake/pipeline_runner.bat.in
      ${VIAME_PIPELINE_RUNNER_SCRIPT}
    )
    configure_file(
      ${CMAKE_SOURCE_DIR}/CMake/processopedia.bat.in
      ${VIAME_PROCESSOPEDIA_SCRIPT}
    )
    configure_file(
      ${CMAKE_SOURCE_DIR}/CMake/plugin_explorer.bat.in
      ${VIAME_PLUGIN_EXPLORER_SCRIPT}
    )

    install( PROGRAMS      ${VIAME_PIPELINE_RUNNER_SCRIPT}
                           ${VIAME_PROCESSOPEDIA_SCRIPT}
                           ${VIAME_PLUGIN_EXPLORER_SCRIPT}
             DESTINATION   ${CMAKE_INSTALL_PREFIX} )
  else()
    set( VIAME_SETUP_SCRIPT_FILE    "${VIAME_BINARY_DIR}/setup_viame.sh" )

    configure_file(
      ${CMAKE_SOURCE_DIR}/CMake/setup_viame.sh.in
      ${VIAME_SETUP_SCRIPT_FILE}
    )

    install( PROGRAMS      ${VIAME_SETUP_SCRIPT_FILE}
             DESTINATION   ${CMAKE_INSTALL_PREFIX} )
  endif()

endif()
