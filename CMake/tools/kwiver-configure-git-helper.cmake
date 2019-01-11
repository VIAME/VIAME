#
# Take in a number of arguments and special arguments ``__OUTPUT_PATH__``,
# ``__SOURCE_PATH__`` and ``__TEMP_PATH__``
#

message(STATUS "Source Path       : '${__SOURCE_PATH__}'")
message(STATUS "Intermediate Path : '${__TEMP_PATH__}'")
message(STATUS "Output Path       : '${__OUTPUT_PATH__}'")
message(STATUS "Project Root      : '${KWIVER_SOURCE_DIR}'")

if(NOT EXISTS "${__SOURCE_PATH__}")
  message(FATAL_ERROR "Source file for configuration did not exist! -> ${__SOURCE_PATH__}")
endif()

find_package(Git)
if (Git_FOUND AND IS_DIRECTORY "${KWIVER_SOURCE_DIR}/.git")
  set(KWIVER_BUILT_FROM_GIT TRUE)

  execute_process(
    COMMAND           "${GIT_EXECUTABLE}"
                      rev-parse
                      HEAD
    WORKING_DIRECTORY "${KWIVER_SOURCE_DIR}"
    RESULT_VARIABLE   git_return
    OUTPUT_VARIABLE   kwiver_git_hash)
  execute_process(
    COMMAND           "${GIT_EXECUTABLE}"
                      rev-parse
                      --short
                      HEAD
    WORKING_DIRECTORY "${KWIVER_SOURCE_DIR}"
    RESULT_VARIABLE   git_return
    OUTPUT_VARIABLE   kwiver_git_hash_short)
  execute_process(
    COMMAND           "${GIT_EXECUTABLE}"
                      diff
                      --no-ext-diff
                      --quiet
                      --exit-code
    WORKING_DIRECTORY "${KWIVER_SOURCE_DIR}"
    RESULT_VARIABLE   git_return)

  string(STRIP "${kwiver_git_hash}" kwiver_git_hash)
  string(STRIP "${kwiver_git_hash_short}" kwiver_git_hash_short)

  if (git_return)
    set(kwiver_git_dirty "dirty")
  endif ()

  message(STATUS "version: ${KWIVER_VERSION}")
  message(STATUS "git hash: ${kwiver_git_hash}")
  message(STATUS "git short hash: ${kwiver_git_hash_short}")
  message(STATUS "git dirty: ${kwiver_git_dirty}")
endif ()

# There are TWO configures here on purpose. The second configure containing
# the COPYONLY flag, only copies the file if the source and dest file are
# different (equivalent to ``cmake -E copy_if_different``). This helps prevent
# files from be touched during a forced configuration when none of the
# contained information changed (prevents rebuilding of dependant targets).
configure_file(
  "${__SOURCE_PATH__}"
  "${__TEMP_PATH__}"
  @ONLY
  )
configure_file(
  "${__TEMP_PATH__}"
  "${__OUTPUT_PATH__}"
  COPYONLY
  )

