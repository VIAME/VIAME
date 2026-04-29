
option( VIAME_BUILD_DIVE_FROM_SOURCE
  "Build DIVE desktop client from source instead of downloading binaries" OFF )
mark_as_advanced( VIAME_BUILD_DIVE_FROM_SOURCE )

set( VIAME_DIVE_BUILD_DIR "${VIAME_BUILD_PREFIX}/src/dive-build" )
set( VIAME_DIVE_INSTALL_DIR "${VIAME_BUILD_INSTALL_PREFIX}/dive" )

file( MAKE_DIRECTORY ${VIAME_DIVE_BUILD_DIR} )

if( VIAME_BUILD_DIVE_FROM_SOURCE )

  set( NODE_EXECUTABLE "" )
  set( NODE_BIN_DIR "" )

  # DIVE requires Node.js >= 22 (see packages/dive/client/.nvmrc)
  set( DIVE_MIN_NODE_VERSION 22 )

  # First check if system node is >= DIVE_MIN_NODE_VERSION
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
      if( SYSTEM_NODE_VERSION_MAJOR GREATER_EQUAL ${DIVE_MIN_NODE_VERSION} )
        set( NODE_EXECUTABLE ${SYSTEM_NODE_EXECUTABLE} )
        get_filename_component( NODE_BIN_DIR ${NODE_EXECUTABLE} DIRECTORY )
        message( STATUS "Found system Node.js ${NODE_VERSION_OUTPUT}" )
      else()
        message( STATUS "System Node.js ${NODE_VERSION_OUTPUT} is < ${DIVE_MIN_NODE_VERSION}, searching for nvm installation..." )
      endif()
    endif()
  endif()

  # If system node is not suitable, look for nvm-installed Node >= DIVE_MIN_NODE_VERSION
  if( NOT NODE_EXECUTABLE AND NOT WIN32 )
    file( GLOB NVM_NODE_DIRS "$ENV{HOME}/.nvm/versions/node/v*" )
    set( BEST_NODE_VERSION 0 )
    set( BEST_NODE_PATH "" )

    foreach( NVM_NODE_DIR ${NVM_NODE_DIRS} )
      get_filename_component( VERSION_DIR_NAME ${NVM_NODE_DIR} NAME )
      string( REGEX MATCH "v([0-9]+)" VERSION_MATCH "${VERSION_DIR_NAME}" )
      set( NVM_NODE_MAJOR "${CMAKE_MATCH_1}" )

      if( NVM_NODE_MAJOR GREATER_EQUAL ${DIVE_MIN_NODE_VERSION} )
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
    message( FATAL_ERROR "VIAME_BUILD_DIVE_FROM_SOURCE requires Node.js >= ${DIVE_MIN_NODE_VERSION} but none was found. "
      "Please install Node.js >= ${DIVE_MIN_NODE_VERSION} or use nvm to install it." )
  endif()

  # DIVE switched from yarn to npm in v1.10+. Locate the npm CLI that ships
  # with the chosen Node.js installation.
  if( WIN32 )
    set( NPM_NAMES npm.cmd npm.ps1 npm )
  else()
    set( NPM_NAMES npm )
  endif()

  set( NPM_EXECUTABLE "" )
  if( NODE_BIN_DIR )
    foreach( _name ${NPM_NAMES} )
      if( EXISTS "${NODE_BIN_DIR}/${_name}" )
        set( NPM_EXECUTABLE "${NODE_BIN_DIR}/${_name}" )
        break()
      endif()
    endforeach()
  endif()
  if( NOT NPM_EXECUTABLE )
    find_program( NPM_EXECUTABLE NAMES ${NPM_NAMES} )
  endif()

  if( NOT NPM_EXECUTABLE )
    message( FATAL_ERROR "VIAME_BUILD_DIVE_FROM_SOURCE requires npm but it was not found. "
      "npm ships with Node.js — verify your Node.js install is complete." )
  endif()

  execute_process(
    COMMAND ${NPM_EXECUTABLE} --version
    OUTPUT_VARIABLE NPM_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NPM_VERSION_RESULT
  )

  if( NPM_VERSION_RESULT EQUAL 0 )
    message( STATUS "Found npm ${NPM_VERSION_OUTPUT} at ${NPM_EXECUTABLE}" )
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
    # Add safe.directory to avoid "dubious ownership" errors in Docker/CI
    execute_process(
      COMMAND git config --global --add safe.directory "${VIAME_PACKAGES_DIR}/dive"
      WORKING_DIRECTORY "${VIAME_PACKAGES_DIR}/dive"
      ERROR_QUIET
    )
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

  # Prepend the node bin dir to PATH for the build commands so npm uses the correct node
  # On Windows, skip PATH modification as CMake interprets semicolons as list separators
  # and Node.js is typically already in the system PATH
  if( NODE_BIN_DIR AND NOT WIN32 )
    set( DIVE_BUILD_ENV ${CMAKE_COMMAND} -E env "PATH=${NODE_BIN_DIR}:$ENV{PATH}" )
  else()
    set( DIVE_BUILD_ENV "" )
  endif()

  # `npm ci` is faster and stricter than `npm install` (uses package-lock.json
  # exactly), but falls back to `npm install` if package-lock.json is missing.
  if( EXISTS "${DIVE_CLIENT_DIR}/package-lock.json" )
    set( DIVE_INSTALL_CMD ${DIVE_BUILD_ENV} ${NPM_EXECUTABLE} ci )
  else()
    set( DIVE_INSTALL_CMD ${DIVE_BUILD_ENV} ${NPM_EXECUTABLE} install )
  endif()

  ExternalProject_Add( dive
    PREFIX ${VIAME_BUILD_PREFIX}
    SOURCE_DIR ${DIVE_CLIENT_DIR}
    BUILD_IN_SOURCE 1
    USES_TERMINAL_BUILD 1
    CONFIGURE_COMMAND ${DIVE_INSTALL_CMD}
    BUILD_COMMAND ${DIVE_BUILD_ENV} ${NPM_EXECUTABLE} run build:electron
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
