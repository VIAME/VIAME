
function( FormatPassdowns _str _varResult )
  set( _tmpResult "" )
  get_cmake_property( _vars VARIABLES )
  string( REGEX MATCHALL "(^|;)${_str}[A-Za-z0-9_]*" _matchedVars "${_vars}" )
  foreach( _match ${_matchedVars} )
    set( _tmpResult ${_tmpResult} "-D${_match}=${${_match}}" )
  endforeach()
  set( ${_varResult} ${_tmpResult} PARENT_SCOPE )
endfunction()

function( DownloadFile _URL _OutputLoc _MD5 )
  message( STATUS "Downloading data file from ${_URL}" )
  file( DOWNLOAD ${_URL} ${_OutputLoc} EXPECTED_MD5 ${_MD5} )
endfunction()

function( DownloadAndExtract _URL _MD5 _DL_LOC _EXT_LOC )
  DownloadFile( ${_URL} ${_DL_LOC} ${_MD5} )
  message( STATUS "Extracting data file from ${_URL}" )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${_DL_LOC}
    WORKING_DIRECTORY ${_EXT_LOC} )
endfunction()

function( DownloadExtractAndInstall _URL _MD5 _DL_LOC _EXT_LOC _INT_LOC )
  DownloadAndExtract( ${_URL} ${_MD5} ${_DL_LOC} ${_EXT_LOC} )
  foreach( _file ${ARGN} )
    if( NOT EXISTS "${_EXT_LOC}/${_file}" )
      message( FATAL_ERROR "${_EXT_LOC}/${_file} does not exist" )
    endif()
    if( IS_DIRECTORY "${_EXT_LOC}/${_file}"  )
      install( DIRECTORY ${ARGN} DESTINATION ${_INT_LOC} )
    else()
      install( FILES ${ARGN} DESTINATION ${_INT_LOC} )
    endif()
  endforeach()
endfunction()

function( RenameSubstr _fnRegex _inStr _outStr )
  file( GLOB DIR_FILES ${_fnRegex} )
  foreach( FN ${DIR_FILES} )
    get_filename_component( FN_WO_DIR ${FN} NAME )
    get_filename_component( FN_DIR ${FN} DIRECTORY )
    string( REPLACE "${_inStr}" "${_outStr}" NEW_FN ${FN_WO_DIR} )
    file( RENAME "${FN}" "${FN_DIR}/${NEW_FN}" )
  endforeach( FN )
endfunction()

function( CopyFiles _inRegex _outDir )
  file( GLOB FILES_TO_COPY ${_inRegex} )
  if( FILES_TO_COPY )
    file( COPY ${FILES_TO_COPY} DESTINATION ${_outDir} )
  endif()
endfunction()

function( RemoveDir _inDir )
  file( REMOVE_RECURSE ${_inDir} )
endfunction()


###
# Removes common indentation from a block of text to produce code suitable for
# setting to `python -c`, or using with pycmd. This allows multiline code to be
# nested nicely in the surrounding code structure.
#
# This function respsects PYTHON_EXECUTABLE if it defined, otherwise it uses
# `python` and hopes for the best. An error will be thrown if it is not found.
#
# Args:
#     outvar : variable that will hold the stdout of the python command
#     text   : text to remove indentation from
#
function(dedent outvar text)
  # Use PYTHON_EXECUTABLE if it is defined, otherwise default to python
  if ("${PYTHON_EXECUTABLE}" STREQUAL "")
    set(_python_exe "python")
  else()
    set(_python_exe "${PYTHON_EXECUTABLE}")
  endif()
  set(_fixup_cmd "import sys; from textwrap import dedent; print(dedent(sys.stdin.read()))")
  # Use echo to pipe the text to python's stdinput. This prevents us from
  # needing to worry about any sort of special escaping.
  execute_process(
    COMMAND echo "${text}"
    COMMAND "${_python_exe}" -c "${_fixup_cmd}"
    RESULT_VARIABLE _dedent_exitcode
    OUTPUT_VARIABLE _dedent_text)
  if(NOT ${_dedent_exitcode} EQUAL 0)
    message(ERROR " Failed to remove indentation from: \n\"\"\"\n${text}\n\"\"\"
    Python dedent failed with error code: ${_dedent_exitcode}")
    message(FATAL_ERROR " Python dedent failed with error code: ${_dedent_exitcode}")
  endif()
  # Remove supurflous newlines (artifacts of print)
  string(STRIP "${_dedent_text}" _dedent_text)
  set(${outvar} "${_dedent_text}" PARENT_SCOPE)
endfunction()


###
# Helper function to run `python -c "<cmd>"` and capture the results of stdout
#
# Runs a python command and populates an outvar with the result of stdout.
# Common indentation in the text of `cmd` is removed before the command is
# executed, so the caller does not need to worry about indentation issues.
#
# This function respsects PYTHON_EXECUTABLE if it defined, otherwise it uses
# `python` and hopes for the best. An error will be thrown if it is not found.
#
# Args:
#     outvar : variable that will hold the stdout of the python command
#     cmd    : text representing a (possibly multiline) block of python code
#
function(pycmd outvar cmd)
  dedent(_dedent_cmd "${cmd}")
  # Use PYTHON_EXECUTABLE if it is defined, otherwise default to python
  if ("${PYTHON_EXECUTABLE}" STREQUAL "")
    set(_python_exe "python")
  else()
    set(_python_exe "${PYTHON_EXECUTABLE}")
  endif()
  # run the actual command
  execute_process(
    COMMAND "${_python_exe}" -c "${_dedent_cmd}"
    RESULT_VARIABLE _exitcode
    OUTPUT_VARIABLE _output)
  if(NOT ${_exitcode} EQUAL 0)
    message(ERROR " Failed when running python code: \"\"\"\n${_dedent_cmd}\n\"\"\"")
    message(FATAL_ERROR " Python command failed with error code: ${_exitcode}")
  endif()
  # Remove supurflous newlines (artifacts of print)
  string(STRIP "${_output}" _output)
  set(${outvar} "${_output}" PARENT_SCOPE)
endfunction()
