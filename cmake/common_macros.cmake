
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
# Adds a standard VIAME forcebuild custom step to an external project
###
function( VIAME_ExternalProject_Add_Step_Forcebuild target_name )
  cmake_parse_arguments(MY "FORCE" "" "" "${ARGN}")

  string(TOUPPER "${target_name}" target_name_upper)
  set (_forcebuild_varname "VIAME_FORCEBUILD_${target_name_upper}")
  set (_forcebuild_value ${${_forcebuild_varname}})

  if (NOT "${MY_FORCE}")
    if ("${_forcebuild_value}" STREQUAL "")
      message(WARNING "${_forcebuild_varname} is not set")
      set(_forcebuild_value True)
    endif()
  endif()

  if(MY_FORCE OR _forcebuild_value)
    ExternalProject_Add_Step(${target_name} forcebuild
      COMMAND ${CMAKE_COMMAND}
        -E remove ${VIAME_BUILD_PREFIX}/src/${target_name}-stamp/${target_name}-build
      COMMENT "Removing ${target_name} build stamp file for build update (forcebuild)."
      DEPENDEES configure
      DEPENDERS build
      ALWAYS 1
      )
  endif()
endfunction()
