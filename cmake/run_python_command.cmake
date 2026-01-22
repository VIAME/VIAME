# CMake script to run a command with environment variables
#
# Parameters:
#   COMMAND_TO_RUN - Command and arguments separated by ----
#   ENV_VARS - Environment variables separated by ---- (each in NAME=VALUE format)
#   WORKING_DIR - Optional working directory for the command
#
# Note: On Windows, path values use <PS> as a placeholder for the path separator
# (;) since semicolons conflict with CMake list separators. This script converts
# <PS> back to the native path separator.

# Convert ----separated command string back to a list
string( REPLACE "----" ";" _command "${COMMAND_TO_RUN}" )

# Unset PYTHONHOME by default to avoid interference from system environment
# If the caller wants to set PYTHONHOME, they can include it in ENV_VARS
# This prevents the "No module named 'encodings'" error that occurs when
# the system PYTHONHOME points to a different Python installation
unset( ENV{PYTHONHOME} )

# Set up environment variables if provided
if( ENV_VARS )
  string( REPLACE "----" ";" _env_list "${ENV_VARS}" )
  foreach( _env_var IN LISTS _env_list )
    if( _env_var )
      string( REGEX MATCH "^([^=]+)=(.*)$" _match "${_env_var}" )
      if( _match )
        set( _env_name "${CMAKE_MATCH_1}" )
        set( _env_value "${CMAKE_MATCH_2}" )
        # Convert <PS> placeholder back to native path separator
        if( WIN32 )
          string( REPLACE "<PS>" ";" _env_value "${_env_value}" )
        else()
          string( REPLACE "<PS>" ":" _env_value "${_env_value}" )
        endif()
        set( ENV{${_env_name}} "${_env_value}" )
      endif()
    endif()
  endforeach()
endif()

# Execute the command
if( WORKING_DIR )
  execute_process(
    COMMAND ${_command}
    RESULT_VARIABLE _result
    WORKING_DIRECTORY ${WORKING_DIR}
    )
else()
  execute_process(
    COMMAND ${_command}
    RESULT_VARIABLE _result
    )
endif()

if( NOT _result EQUAL 0 )
  message( FATAL_ERROR "Command exited with non-zero status: ${_result}" )
endif()
