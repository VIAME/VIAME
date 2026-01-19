# CMake script to run pip install with file locking to prevent race conditions
#
# Parameters (choose one mode):
#   Mode 1 - Wheel install:
#     WHEEL_DIR - Directory containing .whl files to install
#
#   Mode 2 - Direct args:
#     PIP_ARGS - Arguments to pass to pip install (separated by ----)
#
# Common parameters:
#   Python_EXECUTABLE - Path to python executable
#   WORKING_DIR - Optional working directory
#   NO_CACHE_DIR - If set to TRUE, adds --no-cache-dir to pip command

cmake_minimum_required( VERSION 3.16 )

# Build the pip install arguments
if( WHEEL_DIR )
  # Mode 1: Install wheels from directory
  file( GLOB _pip_args LIST_DIRECTORIES FALSE ${WHEEL_DIR}/*.whl )
  set( _working_dir "${WHEEL_DIR}" )
elseif( PIP_ARGS )
  # Mode 2: Use provided arguments
  string( REPLACE "----" ";" _pip_args "${PIP_ARGS}" )
  set( _working_dir "${WORKING_DIR}" )
else()
  message( FATAL_ERROR "pip_install_with_lock.cmake requires either WHEEL_DIR or PIP_ARGS" )
endif()

# Use /tmp for lock file to ensure all parallel pip installs use the same lock
set( _lock_file "/tmp/viame_pip_install.lock" )

# Build cache flag
if( NO_CACHE_DIR )
  set( _cache_flag "--no-cache-dir" )
else()
  set( _cache_flag "" )
endif()

if( UNIX )
  # Use flock on Unix to serialize pip installs (5 minute timeout)
  if( _working_dir )
    execute_process(
      COMMAND flock --timeout 300 ${_lock_file}
        ${Python_EXECUTABLE} -m pip install --no-build-isolation --user ${_cache_flag} ${_pip_args}
      RESULT_VARIABLE _result
      WORKING_DIRECTORY ${_working_dir}
    )
  else()
    execute_process(
      COMMAND flock --timeout 300 ${_lock_file}
        ${Python_EXECUTABLE} -m pip install --no-build-isolation --user ${_cache_flag} ${_pip_args}
      RESULT_VARIABLE _result
    )
  endif()
else()
  # On Windows, add retries for race conditions
  set( _max_retries 5 )
  set( _retry_count 0 )
  set( _result 1 )

  while( _result AND _retry_count LESS _max_retries )
    if( _working_dir )
      execute_process(
        COMMAND ${Python_EXECUTABLE} -m pip install --no-build-isolation --user ${_cache_flag} ${_pip_args}
        RESULT_VARIABLE _result
        WORKING_DIRECTORY ${_working_dir}
      )
    else()
      execute_process(
        COMMAND ${Python_EXECUTABLE} -m pip install --no-build-isolation --user ${_cache_flag} ${_pip_args}
        RESULT_VARIABLE _result
      )
    endif()

    if( _result )
      math( EXPR _retry_count "${_retry_count} + 1" )
      if( _retry_count LESS _max_retries )
        message( STATUS "pip install failed, retrying (${_retry_count}/${_max_retries})..." )
        execute_process( COMMAND ${CMAKE_COMMAND} -E sleep 5 )
      endif()
    endif()
  endwhile()
endif()

if( NOT _result EQUAL 0 )
  message( FATAL_ERROR "pip install exited with non-zero status: ${_result}" )
endif()
