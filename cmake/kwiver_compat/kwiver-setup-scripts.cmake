###
# Configure setup scripts.
#
# These scripts are provided and configured to be sourced before using the
# build-tree. The setup script may optionally also be installed into the
# install-tree.
#

# Create initial setup shell script
set(KWIVER_SETUP_SCRIPT_FILE    "${KWIVER_BINARY_DIR}/setup_KWIVER.sh" )
# Create initial setup batch script
set(KWIVER_SETUP_BATCH_FILE    "${KWIVER_BINARY_DIR}/setup_KWIVER.bat" )
list(APPEND SETUP_BATCH_FILES "${KWIVER_SETUP_BATCH_FILE}")
# Set the bat to use when setting up a test
set(KWIVER_TEST_BATCH_FILE ${KWIVER_SETUP_BATCH_FILE})
# Create initial setup powershell script
set(KWIVER_SETUP_POWERSHELL_FILE "${KWIVER_BINARY_DIR}/setup_KWIVER.ps1")

set(LIBRARY_PATH_VAR "LD_LIBRARY_PATH")
if( APPLE )
  set(LIBRARY_PATH_VAR "DYLD_FALLBACK_LIBRARY_PATH")
endif()

configure_file(
  ${KWIVER_CMAKE_DIR}/setup_KWIVER.sh.in
  ${KWIVER_SETUP_SCRIPT_FILE}
  @ONLY
  )

# Convert Windows path slashes.
file(TO_NATIVE_PATH "${EXTRA_WIN32_PATH}" EXTRA_WIN32_PATH)
file(TO_NATIVE_PATH "${kwiver_plugin_module_subdir}" kwiver_plugin_module_subdir_win)
file(TO_NATIVE_PATH "${kwiver_plugin_process_subdir}" kwiver_plugin_process_subdir_win)
file(TO_NATIVE_PATH "${kwiver_plugin_algorithms_subdir}" kwiver_plugin_algorithms_subdir_win)

if(KWIVER_ENABLE_CUDA)
  list(APPEND EXTRA_WIN32_PATH "${CUDA_TOOLKIT_ROOT_DIR}/bin")
endif()
if(KWIVER_ENABLE_CUDNN)
  list(APPEND EXTRA_WIN32_PATH "${CUDNN_TOOLKIT_ROOT_DIR}/bin")
endif()
configure_file(
  ${KWIVER_CMAKE_DIR}/setup_KWIVER.bat.in
  ${KWIVER_SETUP_BATCH_FILE}
  @ONLY
  )
configure_file(
  ${KWIVER_CMAKE_DIR}/setup_KWIVER.ps1.in
  ${KWIVER_SETUP_POWERSHELL_FILE}
  @ONLY
  )

# install set up script
option( KWIVER_INSTALL_SET_UP_SCRIPT "Creates a setup_KWIVER script (.sh, .bat, and .ps1) that will add properly add kwiver to a shell/cmd/powershell prompt" ON )
mark_as_advanced( KWIVER_INSTALL_SET_UP_SCRIPT )

if( KWIVER_INSTALL_SET_UP_SCRIPT )
  install( PROGRAMS   ${KWIVER_SETUP_SCRIPT_FILE}
    DESTINATION ${CMAKE_INSTALL_PREFIX} )
  if(WIN32)
    install( PROGRAMS   ${KWIVER_SETUP_BATCH_FILE}
      DESTINATION ${CMAKE_INSTALL_PREFIX} )
    install( PROGRAMS   ${KWIVER_SETUP_POWERSHELL_FILE}
      DESTINATION ${CMAKE_INSTALL_PREFIX} )
  endif()
endif()

if(KWIVER_ENABLE_GDAL)
  find_package( GDAL REQUIRED )
  file(REAL_PATH "${GDAL_INCLUDE_DIRS}/../share/gdal" GDAL_DATA)
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set GDAL_DATA=${GDAL_DATA}\n" )
  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "$ENV:GDAL_DATA = \"${GDAL_DATA}\"\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export GDAL_DATA=${GDAL_DATA}\n")
endif()

if(KWIVER_ENABLE_PROJ)
  find_package( PROJ REQUIRED )
  file(REAL_PATH "${PROJ_INCLUDE_DIR}/../share/proj" PROJ_LIB)
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set PROJ_LIB=${PROJ_LIB}\n" )
  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "$ENV:PROJ_LIB = \"${PROJ_LIB}\"\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export PROJ_LIB=${PROJ_LIB}\n" )
  # install proj.db so it can be included in the python wheel
  install(FILES "${PROJ_LIB}/proj.db"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/share")
endif()



if ( fletch_FOUND )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set PATH=${fletch_ROOT}/bin;%PATH%;\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set PATH=${fletch_ROOT}/x64/${_vcVersion}/bin;%PATH%;\n" )

  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "$ENV:PATH = \"${fletch_ROOT}/bin;$ENV:PATH\"\n" )
  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "$ENV:PATH = \"${fletch_ROOT}/x64/${_vcVersion}/bin;$ENV:PATH\"\n" )

  # Could be handled by rpaths, but still needed if Fletch is not packaged with KWIVER
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export ${LIBRARY_PATH_VAR}=${fletch_ROOT}/lib:$${LIBRARY_PATH_VAR}\n" )
else()
  if(WIN32)
    message(WARNING "set fletch_DIR or use spack, otherwise paths to external libraries will not be set")
  endif()
endif()

###
# Install the basic logger properties file.
file( COPY log4cxx.properties       DESTINATION  "${KWIVER_BINARY_DIR}" )
if( NOT SKBUILD )
  # Doesn't need to be in the python package
  install( FILES log4cxx.properties   DESTINATION ${CMAKE_INSTALL_PREFIX} )
endif()

if ( KWIVER_ENABLE_LOG4CXX )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export VITAL_LOGGER_FACTORY=$this_dir/lib/${kwiver_plugin_logger_subdir}/vital_log4cxx_logger\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export LOG4CXX_CONFIGURATION=$this_dir/log4cxx.properties\n" )
  if(WIN32)
    message(STATUS "Log4CXX is not supported on windows, if no other logger is provided, the default will be used")
  endif()

endif()

###
file( COPY log4cplus.properties       DESTINATION  "${KWIVER_BINARY_DIR}" )
if( NOT SKBUILD )
  # Doesn't need to be in the python package
  install( FILES log4cplus.properties   DESTINATION ${CMAKE_INSTALL_PREFIX} )
endif()

if ( KWIVER_ENABLE_LOG4CPLUS )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export VITAL_LOGGER_FACTORY=$this_dir/lib/${kwiver_plugin_logger_subdir}/vital_log4cplus_logger\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export LOG4CPLUS_CONFIGURATION=$this_dir/log4cplus.properties\n" )

  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set VITAL_LOGGER_FACTORY=vital_log4cplus_logger\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set LOG4CPLUS_CONFIGURATION=%~dp0/log4cplus.properties\n" )

  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "$ENV:VITAL_LOGGER_FACTORY = \"vital_log4cplus_logger\"\n" )
  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "$ENV:LOG4CPLUS_CONFIGURATION = \"$this_dir/log4cplus.properties\"\n" )
endif()

file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export KWIVER_DEFAULT_LOG_LEVEL=WARN\n" )
file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set KWIVER_DEFAULT_LOG_LEVEL=WARN\n" )
file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "$ENV:KWIVER_DEFAULT_LOG_LEVEL = \"WARN\"\n" )

if (KWIVER_ENABLE_PYTHON)
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "# Python environment\n")
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export PYTHON_LIBRARY=\"${Python_LIBRARIES}\"\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export PYTHONPATH=$this_dir/${python_site_packages}:$PYTHONPATH\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "\n# additional python modules to load, separated by ':'\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export SPROKIT_PYTHON_MODULES=")
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "kwiver.sprokit.processes:")
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "kwiver.sprokit.schedulers:")
  if (KWIVER_ENABLE_PYTHON_TESTS)
    file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "kwiver.sprokit.tests.processes:")
    file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "kwiver.vital.tests.alg:")
  endif()
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "kwiver.sprokit.processes.pytorch\n")
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "\n# set to suppress loading python modules/processes\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "# export SPROKIT_NO_PYTHON_MODULES\n\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export KWIVER_PYTHON_DEFAULT_LOG_LEVEL=WARN\n" )


  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" ":: Python environment\n")
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set PYTHON_LIBRARY=${Python_LIBRARIES}\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set PYTHONPATH=%~dp0/${python_site_packages};%pythonpath%\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "\n:: additional python modules to load, separated by ':'\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set SPROKIT_PYTHON_MODULES=" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "kwiver.sprokit.processes:" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "kwiver.sprokit.schedulers:" )
  if (KWIVER_ENABLE_PYTHON_TESTS)
    file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "kwiver.sprokit.tests.processes:")
    file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "kwiver.vital.tests.alg:" )
  endif()
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "kwiver.sprokit.processes.pytorch\n")
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "\n:: set to suppress loading python modules/processes\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "::set SPROKIT_NO_PYTHON_MODULES=false\n\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set KWIVER_PYTHON_DEFAULT_LOG_LEVEL=WARN\n" )


  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "# Python environment\n")
  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "$ENV:PYTHON_LIBRARY = \"${Python_LIBRARIES}\"\n" )
  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "$ENV:PYTHONPATH = \"$this_dir/${python_site_packages};$env:PYTHONPATH\"\n" )
  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "\n# additional python modules to load, separated by ':'\n" )
  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "$ENV:SPROKIT_PYTHON_MODULES = \"" )
  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "kwiver.sprokit.processes:" )
  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "kwiver.sprokit.schedulers:" )
  if (KWIVER_ENABLE_PYTHON_TESTS)
    file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "kwiver.sprokit.tests.processes:")
    file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "kwiver.vital.tests.alg:" )
  endif()
  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "kwiver.sprokit.processes.pytorch\"\n")
  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "\n# set to suppress loading python modules/processes\n" )
  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "#$ENV:SPROKIT_NO_PYTHON_MODULES = \"false\"\n\n" )
  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "$ENV:KWIVER_PYTHON_DEFAULT_LOG_LEVEL = \"WARN\"\n" )
endif()
if (KWIVER_ENABLE_PYTHON_TESTS)
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export KWIVER_PYTHON_PLUGIN_PATH=\"kwiver.vital.tests.alg\"")
  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "$ENV:KWIVER_PYTHON_PLUGIN_PATH=\"kwiver.vital.tests.alg\"")
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set KWIVER_PYTHON_PLUGIN_PATH=\"kwiver.vital.tests.alg\"")
endif()

if ( KWIVER_ENABLE_MATLAB )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export LD_LIBRARY_PATH=${Matlab_LIBRARY_DIR}:$LD_LIBRARY_PATH\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set PATH=${Matlab_LIBRARY_DIR};%PATH%\n" )
  file( APPEND "${KWIVER_SETUP_POWERSHELL_FILE}" "$ENV:PATH = \"${Matlab_LIBRARY_DIR};$ENV:PATH\"\n" )
endif()
