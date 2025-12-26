
function( FormatPassdownsCond _str _varResult _bothCases _ignoreStr )
  set( _tmpResult "" )
  get_cmake_property( _vars VARIABLES )
  string( TOUPPER ${_str} _str_caps )
  string( REGEX MATCHALL "(^|;)${_str}[A-Za-z0-9_-]*" _matchedVars "${_vars}" )
  if( _bothCases AND NOT "${_str}" STREQUAL "${_str_caps}" )
    string( REGEX MATCHALL "(^|;)${_str_caps}[A-Za-z0-9_]*" _matchExt "${_vars}" )
    list( APPEND _matchedVars ${_matchExt} )
  endif()
  foreach( _match ${_matchedVars} )
    set( _var_name ${_match} )
    if( "${_ignoreStr}" AND "${_var_name}" MATCHES "${_ignoreStr}" )
      continue()
    endif()
    get_property( _type_set_in_cache CACHE ${_var_name} PROPERTY TYPE SET )
    set( _var_type "STRING" )
    if( _type_set_in_cache )
      get_property( _var_type CACHE ${_var_name} PROPERTY TYPE )
    endif()
    if( WIN32 )
      string( REPLACE "\\" "/" _adjPath "${${_match}}" )
      set( _tmpResult ${_tmpResult} "-D${_match}:${_var_type}=${_adjPath}" )
    else()
      set( _tmpResult ${_tmpResult} "-D${_match}:${_var_type}=${${_match}}" )
    endif()
  endforeach()
  set( ${_varResult} ${_tmpResult} PARENT_SCOPE )
endfunction()

function( FormatPassdownsPython _varResult )
  # Backwards compatibility for sub-projects which use "PYTHON_" cmake
  # variables and the old find_package( PythonInterp ) commands instead
  # of the newer find Python. CMake fills in other python vars from exec.
  #
  # We don't pass down all PYTHON_* variables due to confusion amongst
  # different values for these fields across OS (some are lists, values).
  set( _tmpResult
    -DPython_EXECUTABLE:PATH=${Python_EXECUTABLE}
    -DPYTHON_EXECUTABLE:PATH=${PYTHON_EXECUTABLE} )

  set( ${_varResult} ${_tmpResult} PARENT_SCOPE )
endfunction()

function( FormatPassdowns _str _varResult )
  # Special Cases
  if( ${_str} MATCHES "^PYTHON" )
    FormatPassdownsPython( _tmpResult )
    set( ${_varResult} ${_tmpResult} PARENT_SCOPE )
  # Default Case
  else()
    FormatPassdownsCond( ${_str} _tmpResult ON "" )
    set( ${_varResult} ${_tmpResult} PARENT_SCOPE )
  endif()
endfunction()

function( FormatPassdownsWithIgnore _str _varResult _ignoreStr )
  FormatPassdownsCond( ${_str} _tmpResult OFF ${_ignoreStr} )
  set( ${_varResult} ${_tmpResult} PARENT_SCOPE )
endfunction()

function( FormatPassdownsCaseSensitive _str _varResult )
  FormatPassdownsCond( ${_str} _tmpResult OFF "" )
  set( ${_varResult} ${_tmpResult} PARENT_SCOPE )
endfunction()

function( CopyVarsToAllCaps _str )
  get_cmake_property( _vars VARIABLES )
  string( REGEX MATCHALL "(^|;)${_str}[A-Za-z0-9_]*" _matchedVars "${_vars}" )
  foreach( _match ${_matchedVars} )
    string( TOUPPER ${_match} _match_all_caps )
    set( ${_match_all_caps} ${${_match}} CACHE INTERNAL "Forced" FORCE )
  endforeach()
endfunction()

function( DownloadFile _URL _OutputLoc _MD5 )
  message( STATUS "Downloading data file from ${_URL}" )
  file( DOWNLOAD ${_URL} ${_OutputLoc} EXPECTED_MD5 ${_MD5} )
endfunction()

function( ExtractFile _FILE_LOC _EXT_LOC )
  message( STATUS "Extracting data from ${_FILE_LOC}" )
  file( MAKE_DIRECTORY ${_EXT_LOC} )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${_FILE_LOC}
    WORKING_DIRECTORY ${_EXT_LOC} )
endfunction()

function( DownloadAndExtract _URL _MD5 _DL_LOC _EXT_LOC )
  DownloadFile( ${_URL} ${_DL_LOC} ${_MD5} )
  ExtractFile( ${_DL_LOC} ${_EXT_LOC} )
endfunction()

function( DownloadExtractAndInstall _URL _MD5 _DL_LOC _EXT_LOC _INT_LOC )
  DownloadAndExtract( ${_URL} ${_MD5} ${_DL_LOC} ${_EXT_LOC} )
  file( MAKE_DIRECTORY ${_EXT_LOC} )
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

function( DownloadAndExtractAtInstall _URL _MD5 _DL_LOC _INT_LOC )
  DownloadFile( ${_URL} ${_DL_LOC} ${_MD5} )
  install( CODE "execute_process( \
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${_DL_LOC} \
    WORKING_DIRECTORY ${_INT_LOC} )" )
endfunction()

function( DownloadAndInstallAddOn _URL _MD5 _DL_LOC )
  if( VIAME_IN_SUPERBUILD )
    DownloadAndExtract( ${_URL} ${_MD5} ${_DL_LOC} ${CMAKE_INSTALL_PREFIX} )
  else()
    DownloadAndExtractAtInstall( ${_URL} ${_MD5} ${_DL_LOC} ${CMAKE_INSTALL_PREFIX} )
  endif()
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

function( MoveFiles _inRegex _outDir )
  file( GLOB FILES_TO_COPY ${_inRegex} )
  if( FILES_TO_COPY )
    file( COPY ${FILES_TO_COPY} DESTINATION ${_outDir} )
    file( REMOVE ${FILES_TO_COPY} )
  endif()
endfunction()

function( CopyFileIfExists _inFile _outFile )
  file( COPY ${_inFile} DESTINATION ${_outFile} )
endfunction()

function( CreateSymlink _inFile _outFile )
  if( NOT EXISTS ${_outFile} )
    execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink ${_inFile} ${_outFile} )
  endif()
endfunction()

function( CreateDirectory _outFolder )
  if( NOT EXISTS ${_outFolder} )
    file( MAKE_DIRECTORY ${_outFolder} )
  endif()
endfunction()

function( RemoveDir _inDir )
  file( REMOVE_RECURSE ${_inDir} )
endfunction()

function( ParseLinuxOSField field retval )
  file( STRINGS /etc/os-release vars )
  set( ${_value} "${field}-NOTFOUND" )
  foreach( var ${vars} )
    if( var MATCHES "^${field}=(.*)" )
      set( _value "${CMAKE_MATCH_1}" )
      # Value may be quoted in single- or double-quotes; strip them
      if( _value MATCHES "^['\"](.*)['\"]\$" )
        set( _value "${CMAKE_MATCH_1}" )
      endif()
      break()
    endif()
  endforeach()
  set( ${retval} "${_value}" PARENT_SCOPE )
endfunction()

function( ReplaceStringInFile ifile oldstr newstr )
  file( READ "${ifile}" FILE_CONTENTS )
  string( REPLACE "${oldstr}" "${newstr}" FILE_CONTENTS "${FILE_CONTENTS}" )
  file( WRITE "${ifile}" "${FILE_CONTENTS}" )
endfunction()

# Remove project CMake stamp file to trigger rebuild.
# This is the traditional method - always rebuilds.
#
# Usage: RemoveProjectCMakeStamp( project_name )
#
function( RemoveProjectCMakeStamp _project_name )
  ExternalProject_Add_Step( ${_project_name} forcebuild
    COMMAND ${CMAKE_COMMAND}
      -E remove ${VIAME_BUILD_PREFIX}/src/${_project_name}-stamp/${_project_name}-build
    COMMENT "Removing build stamp file for build update (forcebuild)."
    DEPENDEES configure
    DEPENDERS build
    ALWAYS 1
    )
endfunction()

# Only rebuild when source hash changes.
# Uses git commit hash to detect changes in submodules/source directories.
#
# Usage: BuildOnHashChangeOnly( project_name source_dir )
#
function( BuildOnHashChangeOnly _project_name _source_dir )
  ExternalProject_Add_Step( ${_project_name} forcebuild
    COMMAND ${CMAKE_COMMAND}
      -DLIB_NAME=${_project_name}
      -DLIB_SOURCE_DIR=${_source_dir}
      -DSTAMP_DIR=${VIAME_BUILD_PREFIX}/src/${_project_name}-stamp
      -DHASH_FILE=${VIAME_BUILD_PREFIX}/src/${_project_name}-source-hash.txt
      -P ${VIAME_CMAKE_DIR}/check_source_changed.cmake
    COMMENT "Checking if ${_project_name} source has changed..."
    DEPENDEES configure
    DEPENDERS build
    ALWAYS 1
    )
endfunction()
