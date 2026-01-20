# custom_build_check_source.cmake
#
# This script checks if a library's source has changed and only triggers
# a rebuild if necessary. It uses git commit hashes when available,
# falling back to directory modification times otherwise.
#
# Required variables:
#   LIB_NAME       - Name of the library (for messaging)
#   LIB_SOURCE_DIR - Path to the library source directory
#   STAMP_DIR      - Path to the stamp directory to remove on change
#   HASH_FILE      - Path to store the source hash for comparison

# Function to get the current source hash
function(get_source_hash SOURCE_DIR OUT_HASH)
  # Try to get git commit hash first (most reliable for submodules)
  execute_process(
    COMMAND git -C "${SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE GIT_HASH
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GIT_RESULT
  )

  if(GIT_RESULT EQUAL 0 AND GIT_HASH)
    # Also check for uncommitted changes in the source directory
    execute_process(
      COMMAND git -C "${SOURCE_DIR}" diff --quiet HEAD
      RESULT_VARIABLE DIFF_RESULT
    )
    if(NOT DIFF_RESULT EQUAL 0)
      # There are uncommitted changes, append a marker
      set(GIT_HASH "${GIT_HASH}-dirty")
    endif()
    set(${OUT_HASH} "${GIT_HASH}" PARENT_SCOPE)
    return()
  endif()

  # Fallback: use file modification times of key files
  # Look for setup.py, pyproject.toml, and source directories
  set(HASH_INPUT "")

  foreach(CHECK_FILE setup.py pyproject.toml setup.cfg)
    if(EXISTS "${SOURCE_DIR}/${CHECK_FILE}")
      file(TIMESTAMP "${SOURCE_DIR}/${CHECK_FILE}" FILE_TIME)
      string(APPEND HASH_INPUT "${CHECK_FILE}:${FILE_TIME};")
    endif()
  endforeach()

  # Check key source directories
  foreach(CHECK_DIR src csrc)
    if(IS_DIRECTORY "${SOURCE_DIR}/${CHECK_DIR}")
      file(GLOB_RECURSE SRC_FILES "${SOURCE_DIR}/${CHECK_DIR}/*")
      list(LENGTH SRC_FILES FILE_COUNT)
      string(APPEND HASH_INPUT "${CHECK_DIR}:${FILE_COUNT};")
    endif()
  endforeach()

  # Create a simple hash from the collected info
  string(MD5 FALLBACK_HASH "${HASH_INPUT}")
  set(${OUT_HASH} "${FALLBACK_HASH}" PARENT_SCOPE)
endfunction()

# Main logic
if(NOT LIB_SOURCE_DIR OR NOT STAMP_DIR OR NOT HASH_FILE OR NOT LIB_NAME)
  message(FATAL_ERROR "custom_build_check_source.cmake requires LIB_NAME, LIB_SOURCE_DIR, STAMP_DIR, and HASH_FILE")
endif()

# Get current source hash
get_source_hash("${LIB_SOURCE_DIR}" CURRENT_HASH)

# Read stored hash if it exists
set(STORED_HASH "")
if(EXISTS "${HASH_FILE}")
  file(READ "${HASH_FILE}" STORED_HASH)
  string(STRIP "${STORED_HASH}" STORED_HASH)
endif()

# Compare hashes
# Always rebuild if source is dirty (has uncommitted changes) since the dirty hash
# doesn't capture the actual content of local modifications
string(FIND "${CURRENT_HASH}" "-dirty" DIRTY_POS)
if(DIRTY_POS GREATER -1)
  message(STATUS "${LIB_NAME}: Source is dirty, forcing rebuild")
  set(FORCE_REBUILD TRUE)
else()
  set(FORCE_REBUILD FALSE)
endif()

if(NOT FORCE_REBUILD AND CURRENT_HASH STREQUAL STORED_HASH)
  message(STATUS "${LIB_NAME}: Source unchanged (${CURRENT_HASH}), skipping rebuild")
  # Touch the build stamp to ensure it's newer than configure stamp
  # This prevents rebuilds when cmake reconfigures but source hasn't changed
  set(BUILD_STAMP "${STAMP_DIR}/${LIB_NAME}-build")
  if(EXISTS "${BUILD_STAMP}")
    file(TOUCH "${BUILD_STAMP}")
  endif()
else()
  if(STORED_HASH)
    message(STATUS "${LIB_NAME}: Source changed (${STORED_HASH} -> ${CURRENT_HASH}), triggering rebuild")
  else()
    message(STATUS "${LIB_NAME}: No previous build hash found, will build")
  endif()

  # Remove only the build stamp to trigger rebuild (not the entire stamp directory)
  # This avoids re-running configure unnecessarily
  set(BUILD_STAMP "${STAMP_DIR}/${LIB_NAME}-build")
  if(EXISTS "${BUILD_STAMP}")
    file(REMOVE "${BUILD_STAMP}")
  endif()

  # Store the new hash
  file(WRITE "${HASH_FILE}" "${CURRENT_HASH}")
endif()
