###
# The VIAME project build
#
# This is the `else()` branch of the top-level `CMakeLists.txt` -- the path
# taken when `VIAME_BUILD_DEPENDENCIES` is OFF, which is to say when
# something else has already built the dependencies and this configure only
# has to build VIAME. The superbuild's `add_project_viame.cmake` re-invoked
# CMake on the same source tree with that option off, so this ran as a
# separate configure with the variables the superbuild chose to pass down.
#
# P1-T01 lifts it out of the branch unchanged, so that the rest of phase 1
# can delete the superbuild around it without the diff being two things at
# once. `include()` rather than `add_subdirectory()`, so the variable scope
# is exactly what it was inside the branch. The only edit is the indent.
#
# `design/STATUS.md`, P1-T01, lists what this file expects to be set for it.
##


include_directories( "${CMAKE_CURRENT_BINARY_DIR}" )

###
# Fletch is not looked for
##
# `find_package( fletch NO_MODULE )` stood here, and put `${fletch_DIR}` on
# `CMAKE_PREFIX_PATH`. By P8 there was nothing left for it to find: VXL went
# in P3, FFmpeg in P4, Eigen in P6, OpenCV's C++ in P7, and zlib, tinyxml,
# libsvm, pybind11, darknet and GoogleTest are `library/tpl/`. What it was
# still doing was defining variables -- `pybind11_INCLUDE_DIRS` among them,
# which was putting the whole of the reference superbuild's `include/` on
# one target's command line, above the vendored headers.
#
# Removing it is the point of phase 1: the build takes nothing from fletch.

###
# OpenCV is not required
#
# `find_package( OpenCV REQUIRED )` stood here, with a version check
# feeding `KWIVER_OPENCV_VERSION_MAJOR` to the 2-versus-3 branches the
# code imported from `arrows/ocv` carried. P7 replaced all of it: nothing
# in VIAME's C++ includes an OpenCV header any more, and that macro has no
# readers left.
#
# `VIAME_ENABLE_OPENCV` stays, and now means what it had quietly come to
# mean: cv2 in python. It gates the `viame` applets that import it --
# calibrate, depth, disparity, mosaic, register, 3d, rectify -- the
# pipelines and examples that use them, and the wheel in the python
# requirements. The user's call, recorded in `design/STATUS.md`: a wheel is
# not a build dependency.
##

###
# The imported kwiver code
##
# P5-T05 dissolved `packages/kwiver`: `library/`, `python/kwiver` and
# `tools/` build what it used to. `cmake/kwiver_compat/` holds its macros,
# and the settings the imported code still needs are in
# `kwiver_settings.cmake` below, after the macros are included.
set( KWIVER_ENABLE_PYTHON  ${VIAME_ENABLE_PYTHON}  )
set( KWIVER_ENABLE_SPROKIT ON )
set( KWIVER_ENABLE_TOOLS   ON )
set( KWIVER_BUILD_SHARED   ${BUILD_SHARED_LIBS} )

link_directories( "${VIAME_BINARY_DIR}/lib" )

# `VIAME_DEPENDENCY_INCLUDE_DIRS` stood here: the install prefix, for the
# plugins that took darknet's and libsvm's headers from it rather than
# from a target. Both are vendored now and carry their own include
# directories, and the prefix was actively harmful -- fletch's `svm.h`
# sits in it, so the vendored copy was being shadowed by the one it had
# replaced.

###
# Vendored third party
#
# Small enough to carry, and carried minimally -- only the files VIAME
# compiles or includes, never a distribution. `library/tpl/README.md`
# says what was left behind in each case and why.
#
# This is P1-T03 arriving ahead of the rest of phase 1: it does not need
# the new top-level CMakeLists, and every package that moves here is one
# fletch no longer has to build.
##
add_subdirectory( "${VIAME_SOURCE_DIR}/library/tpl/tinyxml" library/tpl/tinyxml )
add_subdirectory( "${VIAME_SOURCE_DIR}/library/tpl/miniz" library/tpl/miniz )
add_subdirectory( "${VIAME_SOURCE_DIR}/library/tpl/darknet" library/tpl/darknet )

if( VIAME_ENABLE_TESTS )
  add_subdirectory( "${VIAME_SOURCE_DIR}/library/tpl/googletest"
                    library/tpl/googletest )
endif()

# Builds nothing unless `VIAME_BUILD_PYTHON_FROM_SOURCE` is on, which is
# for a packaged build that cannot assume a python on the target machine
add_subdirectory( "${VIAME_SOURCE_DIR}/library/tpl/cpython"
                  library/tpl/cpython )

if( VIAME_ENABLE_PYTHON )
  add_subdirectory( "${VIAME_SOURCE_DIR}/library/tpl/pybind11"
                    library/tpl/pybind11 )
endif()

if( VIAME_ENABLE_SVM )
  add_subdirectory( "${VIAME_SOURCE_DIR}/library/tpl/libsvm" library/tpl/libsvm )
endif()


# VIAME's own CMake helpers, which replaced kwiver's in P8-T08.
set( VIAME_CMAKE_HELPER_DIR "${VIAME_SOURCE_DIR}/cmake/viame/tools" )
include( "${VIAME_SOURCE_DIR}/cmake/viame/viame-configure.cmake" )
include( "${VIAME_SOURCE_DIR}/cmake/viame/viame-flags-check.cmake" )
include( "${VIAME_SOURCE_DIR}/cmake/viame/viame-targets.cmake" )
include( "${VIAME_SOURCE_DIR}/cmake/viame/viame-tests.cmake" )
if( VIAME_ENABLE_PYTHON )
  include( "${VIAME_SOURCE_DIR}/cmake/viame/viame-setup-python.cmake" )
  include( "${VIAME_SOURCE_DIR}/cmake/viame/viame-python.cmake" )
endif()

# After the macros: `kwiver-utils` pulls in `kwiver-setup-python`, which
# computes an output path of its own from KWIVER_BINARY_DIR.
if( VIAME_ENABLE_PYTHON )
  set( kwiver_python_subdir "${VIAME_PYTHON_STRING}" )
  set( kwiver_python_output_path "${VIAME_BUILD_PREFIX}/${kwiver_python_subdir}" )

  # Two names for two different things, which used to be one name for
  # both. `kwiver_python_install_path` is where kwiver's macros put a
  # package -- `${it}/viame/video_io/x.py` -- so it has to be the
  # site-packages directory itself; `kwiver_python_output_path` already
  # builds into `${it}/${python_sitename}/...` and the two have to agree.
  # `viame_python_install_path` is the directory above, which is what
  # VIAME's own callers append `site-packages` to.
  #
  # They were the same before, which put every VIAME python module one
  # directory above site-packages, where nothing imports it. It worked
  # only because an older install had left a copy in the right place.
  set( viame_python_install_path
    "${VIAME_BUILD_INSTALL_PREFIX}/lib/${kwiver_python_subdir}" )
  set( kwiver_python_install_path
    "${viame_python_install_path}/${python_sitename}" )
endif()

###
# System specific compiler flags
##
# `kwiver_warnings` is a global property, and kwiver's own configure check
# has just filled it with kwiver's list, which is stricter than VIAME's --
# `-Werror=non-virtual-dtor` and `-Werror=zero-as-null-pointer-constant`
# among them. VIAME compiled against an installed kwiver never saw those.
# Start from empty so that `viame-flags` yields VIAME's own set.
set_property( GLOBAL PROPERTY kwiver_warnings )

include( viame-flags )

##
# check compiler support
include( "${VIAME_SOURCE_DIR}/cmake/viame/viame-configcheck.cmake" )

# set the name for our package exports and plugin directories
set( viame_export_name                            viame_exports )
set( kwiver_export_name                           viame_exports )

set( kwiver_plugin_subdir                         viame )
set( kwiver_plugin_process_subdir                 ${kwiver_plugin_subdir}/processes )
set( kwiver_plugin_process_instrumentation_subdir ${kwiver_plugin_subdir}/modules )
set( kwiver_plugin_algorithm_subdir               ${kwiver_plugin_subdir}/modules )
set( kwiver_plugin_scheduler_subdir               ${kwiver_plugin_subdir}/processes )
set( kwiver_plugin_module_subdir                  ${kwiver_plugin_subdir}/modules )
set( kwiver_plugin_plugin_explorer_subdir         ${kwiver_plugin_subdir}/modules/plugin_explorer )
set( kwiver_plugin_logger_subdir                  ${kwiver_plugin_subdir}/modules )
# Kwiver set this one itself; the applets imported in P5-T05 -- sprokit's
# runner, pipe-config and pipe-to-dot, and vital's config explorer -- are
# built in VIAME's scope now and would otherwise land in `lib/` itself.
set( kwiver_plugin_applets_subdir                 ${kwiver_plugin_subdir}/applets )

# `linux-remove-duplicate-cvs` stood here. It deleted the `cv2` wheel from
# site-packages whenever fletch's OpenCV had installed a `cv2*.so` beside it,
# because two cv2 modules in one directory is a coin toss. Fletch does not
# build OpenCV for VIAME any more, so there is no second cv2 to lose to --
# and P1-T08 makes the wheel the only one there is.

# Wants the macros and the plugin subdirectories above it, and everything
# it configures below it.
include( "${VIAME_SOURCE_DIR}/cmake/viame/viame-settings.cmake" )

###
# DIVE
##
include( "${VIAME_SOURCE_DIR}/cmake/viame_dive.cmake" )

###
# The python dependencies
##
# Before `python/`, so that a build which installs them has them in place
# before anything imports one.
if( VIAME_ENABLE_PYTHON )
  include( "${VIAME_SOURCE_DIR}/cmake/viame_python_deps.cmake" )
  include( "${VIAME_SOURCE_DIR}/cmake/viame_python_forks.cmake" )
endif()

###
# Add VIAME subdirs
##
# Tests are registered beside the code they test, so testing and the pytest
# helper have to exist before the libraries are added.
if( VIAME_ENABLE_TESTS )
  enable_testing()
  include( CTest )
  include( "${VIAME_SOURCE_DIR}/tests/common/add_pytest_test.cmake" )
endif()

add_subdirectory( library )

if( VIAME_ENABLE_PYTHON )
  add_subdirectory( python )
endif()

add_subdirectory( examples )
add_subdirectory( configs )
add_subdirectory( tools )

###
# The generated static registry
#
# Last, because it names every library that called
# `viame_register_statically` -- and `tools/` registers the applets, so it
# has to be walked first. The executables that link `viame::registry` are
# above this line; CMake resolves a namespaced target at generate time, by
# which point this directory has defined it.
##
add_subdirectory(
  "${VIAME_SOURCE_DIR}/library/algorithm_framework/registry"
  library/algorithm_framework/registry )

if( VIAME_ENABLE_TESTS )
  add_subdirectory( tests )
endif()

###
# The config package an out-of-tree plugin builds against
##
# Kwiver installed one of these until P5-T05; this replaces it. The targets
# keep the `kwiver::` namespace they have always had, so a plugin that
# linked `kwiver::vital` still does -- phase 11 is what renames them.
set( viame_cmake_install_dir lib${LIB_SUFFIX}/cmake/viame )

# `viame_add_library` and the rest, for the plugin's own CMakeLists.
include( "${VIAME_SOURCE_DIR}/cmake/viame/viame-install-utils.cmake" )

get_property( viame_libs GLOBAL PROPERTY viame_libraries )
string( REPLACE ";" " " viame_libs "${viame_libs}" )

configure_file(
  "${VIAME_SOURCE_DIR}/cmake/viame-config-install.cmake.in"
  "${VIAME_BINARY_DIR}/viame-config-install.cmake"
  @ONLY
  )

viame_export_targets( "${VIAME_BINARY_DIR}/viame-config-targets.cmake" )

install(
  FILES       "${VIAME_BINARY_DIR}/viame-config-install.cmake"
  DESTINATION "${viame_cmake_install_dir}"
  RENAME      viame-config.cmake
  )

install(
  EXPORT      ${kwiver_export_name}
  NAMESPACE   kwiver::
  DESTINATION "${viame_cmake_install_dir}"
  FILE        viame-config-targets.cmake
  )

###
# Configure setup scripts
##
include( set_setup_script_vars )

if( WIN32 )
  set( VIAME_SETUP_SCRIPT "${VIAME_BINARY_DIR}/setup_viame.bat" )

  if( VIAME_FIXUP_BUNDLE OR VIAME_VERSION_RELEASE )
    configure_file(
      ${VIAME_CMAKE_DIR}/setup_viame.rel.bat.in
      ${VIAME_SETUP_SCRIPT}
      @ONLY
    )
  else()
    configure_file(
      ${VIAME_CMAKE_DIR}/setup_viame.dev.bat.in
      ${VIAME_SETUP_SCRIPT}
      @ONLY
    )
  endif()

  install( PROGRAMS      ${VIAME_SETUP_SCRIPT}
           DESTINATION   ${CMAKE_INSTALL_PREFIX} )

  if( VIAME_ENABLE_DIVE )
    install( PROGRAMS     "${VIAME_CMAKE_DIR}/launch_dive_interface.bat"
             DESTINATION   ${CMAKE_INSTALL_PREFIX} )
  endif()
else()
  set( VIAME_SETUP_SCRIPT_FILE    "${VIAME_BINARY_DIR}/setup_viame.sh" )

  if( VIAME_FIXUP_BUNDLE OR VIAME_VERSION_RELEASE )
    configure_file(
      ${VIAME_CMAKE_DIR}/setup_viame.rel.sh.in
      ${VIAME_SETUP_SCRIPT_FILE}
      @ONLY
    )
  else()
    configure_file(
      ${VIAME_CMAKE_DIR}/setup_viame.dev.sh.in
      ${VIAME_SETUP_SCRIPT_FILE}
      @ONLY
    )
  endif()

  install( PROGRAMS      ${VIAME_SETUP_SCRIPT_FILE}
           DESTINATION   ${CMAKE_INSTALL_PREFIX} )
  install( PROGRAMS      ${VIAME_CMAKE_DIR}/download_viame_addons.sh
           DESTINATION   ${CMAKE_INSTALL_PREFIX}/bin )
  install( PROGRAMS      ${VIAME_CMAKE_DIR}/download_viame_addons.csv
           DESTINATION   ${CMAKE_INSTALL_PREFIX}/bin )
  install( PROGRAMS      ${VIAME_CMAKE_DIR}/filter_non_web_pipelines.sh
           DESTINATION   ${CMAKE_INSTALL_PREFIX}/bin )
  install( PROGRAMS      ${VIAME_CMAKE_DIR}/limit_train_time_for_viame_web.sh
           DESTINATION   ${CMAKE_INSTALL_PREFIX}/bin )
  install( PROGRAMS      ${VIAME_CMAKE_DIR}/viame_train_detector
           DESTINATION   ${CMAKE_INSTALL_PREFIX}/bin )

  if( VIAME_ENABLE_DIVE )
    install( PROGRAMS      "${VIAME_CMAKE_DIR}/launch_dive_interface.sh"
             DESTINATION   ${CMAKE_INSTALL_PREFIX} )
  endif()
endif()

###
# Install system libs if packaging enabled
##
if( VIAME_FIXUP_BUNDLE )
  set( VIAME_RELEASE_NOTES_FILE "RELEASE_NOTES.md" )

  install( PROGRAMS      ${VIAME_RELEASE_NOTES_FILE}
           DESTINATION   ${CMAKE_INSTALL_PREFIX} )

  set( CMAKE_INSTALL_UCRT_LIBRARIES TRUE )
  include( InstallRequiredSystemLibraries )
  if( CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS )
    install( PROGRAMS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS} DESTINATION bin COMPONENT System )
  endif()

  if( VIAME_ENABLE_CUDNN )
    install( FILES ${CUDNN_LIBRARY} DESTINATION lib )
  endif()

  configure_file(
    ${VIAME_CMAKE_DIR}/viame-install-fixup.cmake.in
    ${CMAKE_BINARY_DIR}/viame-install-fixup.cmake
    @ONLY
    )
  install( SCRIPT ${CMAKE_BINARY_DIR}/viame-install-fixup.cmake )
endif()
