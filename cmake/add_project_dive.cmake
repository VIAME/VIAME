
option( VIAME_BUILD_DIVE_FROM_SOURCE
  "Build DIVE desktop client from source instead of downloading binaries" OFF )
mark_as_advanced( VIAME_BUILD_DIVE_FROM_SOURCE )

set( VIAME_DIVE_BUILD_DIR "${VIAME_BUILD_PREFIX}/src/dive-build" )
set( VIAME_DIVE_INSTALL_DIR "${VIAME_BUILD_INSTALL_PREFIX}/dive" )

file( MAKE_DIRECTORY ${VIAME_DIVE_BUILD_DIR} )

if( VIAME_BUILD_DIVE_FROM_SOURCE )

  set( NODE_EXECUTABLE "" )
  set( NODE_BIN_DIR "" )

  # First check if system node is >= 18
  find_program( SYSTEM_NODE_EXECUTABLE node )
  if( SYSTEM_NODE_EXECUTABLE )
    execute_process(
      COMMAND ${SYSTEM_NODE_EXECUTABLE} --version
      OUTPUT_VARIABLE NODE_VERSION_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE NODE_VERSION_RESULT
    )
    if( NODE_VERSION_RESULT EQUAL 0 )
      string( REGEX MATCH "v([0-9]+)" NODE_VERSION_MATCH "${NODE_VERSION_OUTPUT}" )
      set( SYSTEM_NODE_VERSION_MAJOR "${CMAKE_MATCH_1}" )
      if( SYSTEM_NODE_VERSION_MAJOR GREATER_EQUAL 18 )
        set( NODE_EXECUTABLE ${SYSTEM_NODE_EXECUTABLE} )
        get_filename_component( NODE_BIN_DIR ${NODE_EXECUTABLE} DIRECTORY )
        message( STATUS "Found system Node.js ${NODE_VERSION_OUTPUT}" )
      else()
        message( STATUS "System Node.js ${NODE_VERSION_OUTPUT} is < 18, searching for nvm installation..." )
      endif()
    endif()
  endif()

  # If system node is not suitable, look for nvm-installed Node >= 18
  if( NOT NODE_EXECUTABLE AND NOT WIN32 )
    file( GLOB NVM_NODE_DIRS "$ENV{HOME}/.nvm/versions/node/v*" )
    set( BEST_NODE_VERSION 0 )
    set( BEST_NODE_PATH "" )

    foreach( NVM_NODE_DIR ${NVM_NODE_DIRS} )
      get_filename_component( VERSION_DIR_NAME ${NVM_NODE_DIR} NAME )
      string( REGEX MATCH "v([0-9]+)" VERSION_MATCH "${VERSION_DIR_NAME}" )
      set( NVM_NODE_MAJOR "${CMAKE_MATCH_1}" )

      if( NVM_NODE_MAJOR GREATER_EQUAL 18 )
        if( EXISTS "${NVM_NODE_DIR}/bin/node" )
          if( NVM_NODE_MAJOR GREATER BEST_NODE_VERSION )
            set( BEST_NODE_VERSION ${NVM_NODE_MAJOR} )
            set( BEST_NODE_PATH "${NVM_NODE_DIR}/bin/node" )
            set( BEST_NODE_BIN_DIR "${NVM_NODE_DIR}/bin" )
          endif()
        endif()
      endif()
    endforeach()

    if( BEST_NODE_PATH )
      set( NODE_EXECUTABLE ${BEST_NODE_PATH} )
      set( NODE_BIN_DIR ${BEST_NODE_BIN_DIR} )
      execute_process(
        COMMAND ${NODE_EXECUTABLE} --version
        OUTPUT_VARIABLE NODE_VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      message( STATUS "Found nvm Node.js ${NODE_VERSION_OUTPUT} at ${NODE_EXECUTABLE}" )
    endif()
  endif()

  if( NOT NODE_EXECUTABLE )
    message( FATAL_ERROR "VIAME_BUILD_DIVE_FROM_SOURCE requires Node.js >= 18 but none was found. "
      "Please install Node.js >= 18 or use nvm to install it." )
  endif()

  # Check for yarn - first in the same bin dir as node, then system-wide
  if( WIN32 )
    set( YARN_EXECUTABLE yarn.cmd )
  else()
    if( NODE_BIN_DIR AND EXISTS "${NODE_BIN_DIR}/yarn" )
      set( YARN_EXECUTABLE "${NODE_BIN_DIR}/yarn" )
    else()
      find_program( YARN_EXECUTABLE yarn )
    endif()
  endif()

  if( NOT YARN_EXECUTABLE OR NOT EXISTS ${YARN_EXECUTABLE} )
    find_program( YARN_EXECUTABLE yarn )
  endif()

  if( NOT YARN_EXECUTABLE )
    message( FATAL_ERROR "VIAME_BUILD_DIVE_FROM_SOURCE requires yarn but it was not found. "
      "Please install yarn (npm install -g yarn)." )
  endif()

  execute_process(
    COMMAND ${YARN_EXECUTABLE} --version
    OUTPUT_VARIABLE YARN_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE YARN_VERSION_RESULT
  )

  if( YARN_VERSION_RESULT EQUAL 0 )
    message( STATUS "Found yarn ${YARN_VERSION_OUTPUT} at ${YARN_EXECUTABLE}" )
  endif()

  set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} dive )

  set( DIVE_CLIENT_DIR "${VIAME_PACKAGES_DIR}/dive/client" )

  # Detect if we switched from downloading binaries to building from source
  set( DIVE_BUILD_MODE_FILE "${VIAME_BUILD_PREFIX}/src/dive-build-mode.txt" )
  set( DIVE_PREVIOUS_BUILD_MODE "" )
  if( EXISTS "${DIVE_BUILD_MODE_FILE}" )
    file( READ "${DIVE_BUILD_MODE_FILE}" DIVE_PREVIOUS_BUILD_MODE )
    string( STRIP "${DIVE_PREVIOUS_BUILD_MODE}" DIVE_PREVIOUS_BUILD_MODE )
  endif()

  if( NOT "${DIVE_PREVIOUS_BUILD_MODE}" STREQUAL "SOURCE" )
    message( STATUS "DIVE build mode changed to building from source" )
    message( STATUS "Cleaning previous DIVE install directory..." )

    if( EXISTS "${VIAME_DIVE_INSTALL_DIR}" )
      file( REMOVE_RECURSE "${VIAME_DIVE_INSTALL_DIR}" )
    endif()

    file( WRITE "${DIVE_BUILD_MODE_FILE}" "SOURCE" )
  endif()

  # Detect if DIVE submodule hash has changed and clean old build if so
  set( DIVE_HASH_FILE "${VIAME_BUILD_PREFIX}/src/dive-hash.txt" )
  set( DIVE_CURRENT_HASH "" )

  if( EXISTS "${VIAME_PACKAGES_DIR}/dive/.git" )
    execute_process(
      COMMAND git rev-parse HEAD
      WORKING_DIRECTORY "${VIAME_PACKAGES_DIR}/dive"
      OUTPUT_VARIABLE DIVE_CURRENT_HASH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE DIVE_HASH_RESULT
    )
  endif()

  if( DIVE_CURRENT_HASH )
    set( DIVE_PREVIOUS_HASH "" )
    if( EXISTS "${DIVE_HASH_FILE}" )
      file( READ "${DIVE_HASH_FILE}" DIVE_PREVIOUS_HASH )
      string( STRIP "${DIVE_PREVIOUS_HASH}" DIVE_PREVIOUS_HASH )
    endif()

    if( NOT "${DIVE_CURRENT_HASH}" STREQUAL "${DIVE_PREVIOUS_HASH}" )
      message( STATUS "DIVE hash changed from ${DIVE_PREVIOUS_HASH} to ${DIVE_CURRENT_HASH}" )
      message( STATUS "Cleaning previous DIVE build and install directories..." )

      # Clean the electron build output
      if( EXISTS "${DIVE_CLIENT_DIR}/dist_electron" )
        file( REMOVE_RECURSE "${DIVE_CLIENT_DIR}/dist_electron" )
      endif()

      # Clean node_modules to force fresh install
      if( EXISTS "${DIVE_CLIENT_DIR}/node_modules" )
        file( REMOVE_RECURSE "${DIVE_CLIENT_DIR}/node_modules" )
      endif()

      # Clean the install directory
      if( EXISTS "${VIAME_DIVE_INSTALL_DIR}" )
        file( REMOVE_RECURSE "${VIAME_DIVE_INSTALL_DIR}" )
      endif()

      # Clean ExternalProject stamps to force rebuild
      file( GLOB DIVE_STAMP_FILES "${VIAME_BUILD_PREFIX}/src/dive-stamp/*" )
      if( DIVE_STAMP_FILES )
        file( REMOVE ${DIVE_STAMP_FILES} )
      endif()

      # Write new hash
      file( WRITE "${DIVE_HASH_FILE}" "${DIVE_CURRENT_HASH}" )
    endif()
  endif()

  if( WIN32 )
    set( DIVE_ELECTRON_OUTPUT_DIR ${DIVE_CLIENT_DIR}/dist_electron/win-unpacked )
  else()
    set( DIVE_ELECTRON_OUTPUT_DIR ${DIVE_CLIENT_DIR}/dist_electron/linux-unpacked )
  endif()

  # Prepend the node bin dir to PATH for the build commands so yarn uses the correct node
  if( NODE_BIN_DIR )
    set( DIVE_BUILD_ENV ${CMAKE_COMMAND} -E env "PATH=${NODE_BIN_DIR}:$ENV{PATH}" )
  else()
    set( DIVE_BUILD_ENV "" )
  endif()

  ExternalProject_Add( dive
    PREFIX ${VIAME_BUILD_PREFIX}
    SOURCE_DIR ${DIVE_CLIENT_DIR}
    BUILD_IN_SOURCE 1
    USES_TERMINAL_BUILD 1
    CONFIGURE_COMMAND ${DIVE_BUILD_ENV} ${YARN_EXECUTABLE} install
    BUILD_COMMAND ${DIVE_BUILD_ENV} ${YARN_EXECUTABLE} build:electron
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${DIVE_ELECTRON_OUTPUT_DIR}
      ${VIAME_DIVE_INSTALL_DIR}
    INSTALL_DIR ${VIAME_DIVE_INSTALL_DIR}
  )

else()

  # Detect if we switched from building from source to downloading binaries
  set( DIVE_BUILD_MODE_FILE "${VIAME_BUILD_PREFIX}/src/dive-build-mode.txt" )
  set( DIVE_PREVIOUS_BUILD_MODE "" )
  if( EXISTS "${DIVE_BUILD_MODE_FILE}" )
    file( READ "${DIVE_BUILD_MODE_FILE}" DIVE_PREVIOUS_BUILD_MODE )
    string( STRIP "${DIVE_PREVIOUS_BUILD_MODE}" DIVE_PREVIOUS_BUILD_MODE )
  endif()

  if( NOT "${DIVE_PREVIOUS_BUILD_MODE}" STREQUAL "BINARY" )
    message( STATUS "DIVE build mode changed to downloading binaries" )
    message( STATUS "Cleaning previous DIVE install directory..." )

    if( EXISTS "${VIAME_DIVE_INSTALL_DIR}" )
      file( REMOVE_RECURSE "${VIAME_DIVE_INSTALL_DIR}" )
    endif()

    file( WRITE "${DIVE_BUILD_MODE_FILE}" "BINARY" )
  endif()

  if( WIN32 )
    DownloadAndExtract(
      https://github.com/Kitware/dive/releases/download/v1.9.10/DIVE-Desktop-1.9.10.zip
      9177cce0cb8fe0f5dbb66f3bb850bc73
      ${VIAME_DOWNLOAD_DIR}/dive_interface_binaries.zip
      ${VIAME_DIVE_BUILD_DIR} )
  elseif( UNIX )
    DownloadAndExtract(
      https://github.com/Kitware/dive/releases/download/v1.9.10/DIVE-Desktop-1.9.10.tar.gz
      a1b0ed424682e7cb3ed6d8e4037a1ba4
      ${VIAME_DOWNLOAD_DIR}/dive_interface_binaries.tar.gz
      ${VIAME_DIVE_BUILD_DIR} )
  endif()

  file( GLOB DIVE_SUBDIRS RELATIVE ${VIAME_DIVE_BUILD_DIR} ${VIAME_DIVE_BUILD_DIR}/DIVE* )

  if( DIVE_SUBDIRS )
    foreach( SUBDIR ${DIVE_SUBDIRS} )
      if( IS_DIRECTORY ${VIAME_DIVE_BUILD_DIR}/${SUBDIR} )
        file( GLOB ALL_FILES_IN_DIR "${VIAME_DIVE_BUILD_DIR}/${SUBDIR}/*" )
        file( COPY ${ALL_FILES_IN_DIR} DESTINATION ${VIAME_DIVE_INSTALL_DIR} )
      endif()
    endforeach()
  else()
    file( GLOB ALL_FILES_IN_DIR "${VIAME_DIVE_BUILD_DIR}/*" )
    file( COPY ${ALL_FILES_IN_DIR} DESTINATION ${VIAME_DIVE_INSTALL_DIR} )
  endif()

endif()
