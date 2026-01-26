
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

# Generate a runtime git clone-or-pull command for use in ExternalProject_Add.
# Uses runtime checks instead of CMake configure-time checks to handle rebuilds
# correctly. If the directory contains a .git folder, it pulls; otherwise it
# removes any existing directory and clones fresh.
#
# Usage in ExternalProject_Add:
#   GitCloneOrPullCmd( MY_CMD https://github.com/org/repo.git ${TARGET_DIR} )
#   ExternalProject_Add( myproject
#     CONFIGURE_COMMAND ${MY_CMD}
#     ...
#   )
#
# With optional branch:
#   GitCloneOrPullCmd( MY_CMD https://github.com/org/repo.git ${TARGET_DIR} my-branch )
#
function( GitCloneOrPullCmd _output_var _repo_url _target_dir )
  set( _branch "${ARGN}" )

  # Generate a unique script name based on target directory
  string( MD5 _hash "${_target_dir}" )
  set( _script_dir "${CMAKE_BINARY_DIR}/git_scripts" )
  set( _script_path "${_script_dir}/git_clone_or_pull_${_hash}.sh" )

  file( MAKE_DIRECTORY "${_script_dir}" )

  if( _branch )
    set( _clone_cmd "git clone --branch ${_branch} ${_repo_url} ${_target_dir}" )
  else()
    set( _clone_cmd "git clone ${_repo_url} ${_target_dir}" )
  endif()

  file( WRITE "${_script_path}"
"#!/bin/sh
if [ -d \"${_target_dir}/.git\" ]; then
  echo \"Pulling updates in ${_target_dir}\"
  git -C \"${_target_dir}\" pull
else
  if [ -d \"${_target_dir}\" ]; then
    echo \"Removing non-git directory ${_target_dir}\"
    rm -rf \"${_target_dir}\"
  fi
  echo \"Cloning ${_repo_url} to ${_target_dir}\"
  ${_clone_cmd}
fi
" )

  set( ${_output_var} sh "${_script_path}" PARENT_SCOPE )
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
      -P ${VIAME_CMAKE_DIR}/custom_build_check_source.cmake
    COMMENT "Checking if ${_project_name} source has changed..."
    DEPENDEES configure
    DEPENDERS build
    ALWAYS 1
    )
endfunction()

# Parse a CSV line handling quoted fields (fields can contain commas inside quotes)
# Returns the parsed fields in the variable named by _result
function( ParseCSVLine _line _result )
  set( _fields )
  set( _current_field "" )
  set( _in_quotes FALSE )
  string( LENGTH "${_line}" _len )

  math( EXPR _last_idx "${_len} - 1" )

  foreach( _i RANGE 0 ${_last_idx} )
    string( SUBSTRING "${_line}" ${_i} 1 _char )

    if( _char STREQUAL "\"" )
      if( _in_quotes )
        set( _in_quotes FALSE )
      else()
        set( _in_quotes TRUE )
      endif()
    elseif( _char STREQUAL "," AND NOT _in_quotes )
      # End of field - trim whitespace
      string( STRIP "${_current_field}" _current_field )
      list( APPEND _fields "${_current_field}" )
      set( _current_field "" )
    else()
      string( APPEND _current_field "${_char}" )
    endif()
  endforeach()

  # Don't forget the last field
  string( STRIP "${_current_field}" _current_field )
  list( APPEND _fields "${_current_field}" )

  set( ${_result} "${_fields}" PARENT_SCOPE )
endfunction()

# Check if the platform matches the current system
# Returns TRUE/FALSE in _result
function( CheckAddonPlatform _platform _result )
  if( "${_platform}" STREQUAL "ALL-PLATFORMS" )
    set( ${_result} TRUE PARENT_SCOPE )
  elseif( "${_platform}" STREQUAL "LINUX-ONLY" )
    if( UNIX AND NOT APPLE )
      set( ${_result} TRUE PARENT_SCOPE )
    else()
      set( ${_result} FALSE PARENT_SCOPE )
    endif()
  elseif( "${_platform}" STREQUAL "WINDOWS-ONLY" )
    if( WIN32 )
      set( ${_result} TRUE PARENT_SCOPE )
    else()
      set( ${_result} FALSE PARENT_SCOPE )
    endif()
  else()
    message( WARNING "Unknown platform specifier: ${_platform}, defaulting to ALL-PLATFORMS" )
    set( ${_result} TRUE PARENT_SCOPE )
  endif()
endfunction()

# Parse the addons CSV file and declare options for each entry
# Also populates VIAME_ADDON_ENTRIES with the parsed data for later use
# Options are declared as OFF by default and marked as advanced
function( ParseModelDownloadOptions _csv_file )
  if( NOT EXISTS "${_csv_file}" )
    message( FATAL_ERROR "Addon CSV file not found: ${_csv_file}" )
  endif()

  file( STRINGS "${_csv_file}" _lines )

  set( _addon_names )
  set( _addon_count 0 )

  foreach( _line IN LISTS _lines )
    # Skip empty lines
    string( STRIP "${_line}" _line_stripped )
    if( "${_line_stripped}" STREQUAL "" )
      continue()
    endif()

    ParseCSVLine( "${_line}" _fields )

    list( LENGTH _fields _num_fields )
    if( _num_fields LESS 6 )
      message( WARNING "Skipping malformed CSV line (expected 6 fields, got ${_num_fields}): ${_line}" )
      continue()
    endif()

    list( GET _fields 0 _name )
    list( GET _fields 1 _url )
    list( GET _fields 2 _description )
    list( GET _fields 3 _md5 )
    list( GET _fields 4 _platform )
    list( GET _fields 5 _enable_flags )

    # Check if this platform applies
    CheckAddonPlatform( "${_platform}" _platform_matches )

    if( _platform_matches )
      # Declare the option as OFF by default
      option( VIAME_DOWNLOAD_MODELS-${_name} "${_description}" OFF )

      # Mark as advanced
      mark_as_advanced( VIAME_DOWNLOAD_MODELS-${_name} )

      # Store the addon data for later use
      set( VIAME_ADDON_${_name}_URL "${_url}" CACHE INTERNAL "" )
      set( VIAME_ADDON_${_name}_MD5 "${_md5}" CACHE INTERNAL "" )
      set( VIAME_ADDON_${_name}_ENABLE_FLAGS "${_enable_flags}" CACHE INTERNAL "" )

      list( APPEND _addon_names "${_name}" )
      math( EXPR _addon_count "${_addon_count} + 1" )
    endif()
  endforeach()

  set( VIAME_ADDON_NAMES "${_addon_names}" CACHE INTERNAL "List of addon names from CSV" )
  message( STATUS "Parsed ${_addon_count} addon model pack options from CSV" )
endfunction()

# Check that all required enable flags are set for enabled addons
# Call this after all options have been configured
function( ValidateAddonDependencies )
  foreach( _name IN LISTS VIAME_ADDON_NAMES )
    if( VIAME_DOWNLOAD_MODELS-${_name} )
      set( _enable_flags "${VIAME_ADDON_${_name}_ENABLE_FLAGS}" )

      # Parse the comma-separated enable flags
      string( REPLACE "," ";" _flag_list "${_enable_flags}" )

      foreach( _flag IN LISTS _flag_list )
        string( STRIP "${_flag}" _flag )
        if( NOT "${_flag}" STREQUAL "" )
          if( NOT VIAME_ENABLE_${_flag} )
            message( FATAL_ERROR
              "VIAME_DOWNLOAD_MODELS-${_name} requires VIAME_ENABLE_${_flag} to be enabled" )
          endif()
        endif()
      endforeach()
    endif()
  endforeach()
endfunction()

# Download an addon package to the downloads folder, extract it to a temp location,
# and install only the 'models' and 'transforms' directories
function( DownloadAndInstallAddonModels _name )
  if( NOT VIAME_DOWNLOAD_MODELS-${_name} )
    return()
  endif()

  set( _url "${VIAME_ADDON_${_name}_URL}" )
  set( _md5 "${VIAME_ADDON_${_name}_MD5}" )
  set( _dl_file "${VIAME_DOWNLOAD_DIR}/VIAME-${_name}-Models.zip" )
  set( _extract_dir "${CMAKE_BINARY_DIR}/addon-extract/${_name}" )

  # Download the file
  DownloadFile( "${_url}" "${_dl_file}" "${_md5}" )

  # Extract to temporary location
  message( STATUS "Extracting addon ${_name} to ${_extract_dir}" )
  file( MAKE_DIRECTORY "${_extract_dir}" )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf "${_dl_file}"
    WORKING_DIRECTORY "${_extract_dir}" )

  # Find and install models directory if it exists
  file( GLOB _models_dirs "${_extract_dir}/*/models" "${_extract_dir}/models" )
  foreach( _models_dir IN LISTS _models_dirs )
    if( IS_DIRECTORY "${_models_dir}" )
      message( STATUS "Installing models from ${_models_dir}" )
      install( DIRECTORY "${_models_dir}/"
               DESTINATION configs/pipelines/models )
    endif()
  endforeach()

  # Find and install transforms directory if it exists
  file( GLOB _transforms_dirs "${_extract_dir}/*/transforms" "${_extract_dir}/transforms" )
  foreach( _transforms_dir IN LISTS _transforms_dirs )
    if( IS_DIRECTORY "${_transforms_dir}" )
      message( STATUS "Installing transforms from ${_transforms_dir}" )
      install( DIRECTORY "${_transforms_dir}/"
               DESTINATION configs/pipelines/transforms )
    endif()
  endforeach()
endfunction()

# Process all enabled addons - validates dependencies and downloads/installs models
function( ProcessAllAddonModels )
  ValidateAddonDependencies()

  foreach( _name IN LISTS VIAME_ADDON_NAMES )
    DownloadAndInstallAddonModels( "${_name}" )
  endforeach()
endfunction()

# Disable all VIAME_DOWNLOAD_MODELS-* options by setting them to OFF
macro( DisableAllModelDownloads )
  get_cmake_property( _all_vars VARIABLES )
  foreach( _var IN LISTS _all_vars )
    if( _var MATCHES "^VIAME_DOWNLOAD_MODELS-" )
      set( ${_var} OFF CACHE BOOL "Forced off" FORCE )
    endif()
  endforeach()
endmacro()
