###
# Configure setup scripts.

# Create initial setup shell script
set(KWIVER_SETUP_SCRIPT_FILE    "${KWIVER_BINARY_DIR}/setup_KWIVER.sh" )
# Create initial setup batch script
set(KWIVER_SETUP_BATCH_FILE    "${KWIVER_BINARY_DIR}/setup_KWIVER.bat" )
list(APPEND SETUP_BATCH_FILES "${KWIVER_SETUP_BATCH_FILE}")
# Set the bat to use when setting up a test
set(KWIVER_TEST_BATCH_FILE ${KWIVER_SETUP_BATCH_FILE})

set(LIBRARY_PATH_VAR "LD_LIBRARY_PATH")
if( APPLE )
  set(LIBRARY_PATH_VAR "DYLD_FALLBACK_LIBRARY_PATH")
endif()

configure_file(
  ${KWIVER_CMAKE_DIR}/setup_KWIVER.sh.in
  ${KWIVER_SETUP_SCRIPT_FILE}
  @ONLY
  )

if(fletch_BUILT_WITH_CUDA)
  list(APPEND EXTRA_WIN32_PATH "${CUDA_TOOLKIT_ROOT_DIR}/bin")
endif()
if(fletch_BUILT_WITH_CUDNN)
  list(APPEND EXTRA_WIN32_PATH "${CUDNN_TOOLKIT_ROOT_DIR}/bin")
endif()
configure_file(
  ${KWIVER_CMAKE_DIR}/setup_KWIVER.bat.in
  ${KWIVER_SETUP_BATCH_FILE}
  @ONLY
  )

# install set up script
option( KWIVER_INSTALL_SET_UP_SCRIPT "Creates a setup_KWIVER script (.sh and .bat) that will add properly add kwiver to a shell/cmd prompt" ON )
mark_as_advanced( KWIVER_INSTALL_SET_UP_SCRIPT )

if( KWIVER_INSTALL_SET_UP_SCRIPT )
  install( PROGRAMS   ${KWIVER_SETUP_SCRIPT_FILE}
    DESTINATION ${CMAKE_INSTALL_PREFIX} )
  if(WIN32)
    install( PROGRAMS   ${KWIVER_SETUP_BATCH_FILE}
      DESTINATION ${CMAKE_INSTALL_PREFIX} )
  endif()
endif()

if ( fletch_FOUND )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set PATH=${fletch_ROOT}/bin;%PATH%;\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set PATH=${fletch_ROOT}/x64/${_vcVersion}/bin;%PATH%;\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set GDAL_DATA=${GDAL_ROOT}/share/gdal\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set PROJ_LIB=${PROJ4_ROOT}/share/proj\n" )

  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export ${LIBRARY_PATH_VAR}=${fletch_ROOT}/lib:$${LIBRARY_PATH_VAR}\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export GDAL_DATA=${GDAL_ROOT}/share/gdal\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export PROJ_LIB=${PROJ4_ROOT}/share/proj\n" )
endif()

###
# Install the basic logger properties file.
file( COPY log4cxx.properties       DESTINATION  "${KWIVER_BINARY_DIR}" )
install( FILES log4cxx.properties   DESTINATION ${CMAKE_INSTALL_PREFIX} )

if ( KWIVER_ENABLE_LOG4CXX )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export VITAL_LOGGER_FACTORY=$this_dir/lib/${kwiver_plugin_logger_subdir}/vital_log4cxx_logger\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export LOG4CXX_CONFIGURATION=$this_dir/log4cxx.properties\n" )
  if(WIN32)
    message(STATUS "Log4CXX is not supported on windows, if no other logger is provided, the default will be used")
  endif()

endif()

###
file( COPY log4cplus.properties       DESTINATION  "${KWIVER_BINARY_DIR}" )
install( FILES log4cplus.properties   DESTINATION ${CMAKE_INSTALL_PREFIX} )

if ( KWIVER_ENABLE_LOG4CPLUS )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export VITAL_LOGGER_FACTORY=$this_dir/lib/${kwiver_plugin_logger_subdir}/vital_log4cplus_logger\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export LOG4CPLUS_CONFIGURATION=$this_dir/log4cplus.properties\n" )

  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set VITAL_LOGGER_FACTORY=vital_log4cplus_logger\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set LOG4CPLUS_CONFIGURATION=%~dp0/log4cplus.properties\n" )
endif()

file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export KWIVER_DEFAULT_LOG_LEVEL=WARN\n" )
file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set KWIVER_DEFAULT_LOG_LEVEL=WARN\n" )

if (KWIVER_ENABLE_PYTHON)
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "# Python environment\n")
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export PYTHON_LIBRARY=\"${PYTHON_LIBRARY}\"\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export PYTHONPATH=$this_dir/${python_site_packages}:$PYTHONPATH\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "\n# additional python mudules to load, separated by ':'\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export SPROKIT_PYTHON_MODULES=kwiver.sprokit.processes:kwiver.sprokit.schedulers:kwiver.sprokit.tests.processes:kwiver.arrows.python\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "\n# set to suppress loading python modules/processes\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "# export SPROKIT_NO_PYTHON_MODULES\n\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export KWIVER_PYTHON_DEFAULT_LOG_LEVEL=WARN\n" )

  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" ":: Python environment\n")
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set PYTHON_LIBRARY=${PYTHON_LIBRARY}\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set PYTHONPATH=%~dp0/lib/%config%/python2.7/site-packages\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "\n:: additional python mudules to load, separated by ':'\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set SPROKIT_PYTHON_MODULES=kwiver.sprokit.processes:kwiver.sprokit.schedulers:kwiver.sprokit.tests.processes:kwiver.arrows.python\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "\n:: set to suppress loading python modules/processes\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "::set SPROKIT_NO_PYTHON_MODULES=false\n\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set KWIVER_PYTHON_DEFAULT_LOG_LEVEL=WARN\n" )
endif()

if ( KWIVER_ENABLE_MATLAB )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export LD_LIBRARY_PATH=${Matlab_LIBRARY_DIR}:$LD_LIBRARY_PATH\n" )

  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set PATH=${Matlab_LIBRARY_DIR};%PATH%\n" )
endif()
