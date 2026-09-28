# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

###
# Apply one unified diff to a fork's source tree, idempotently.
#
# `packages/patches/<fork>/` is the other way VIAME changes a vendored
# package: whole files copied over the source. That is the right shape when a
# file is replaced, and the wrong one when three lines of a nine hundred line
# module change -- it puts a copy of somebody else's file in our repository
# and the copy goes stale silently. A diff says what changed, and stops the
# build when upstream moves under it.
#
# Idempotent because a fork's source is a checked out submodule, not a fresh
# extract: the build may run twice without the tree being reset. `git apply
# --reverse --check` succeeding means the patch is already in, which is not an
# error.
#
# Arguments, all required:
#   PATCH_FILE  the unified diff, paths relative to the fork root (-p1)
#   SOURCE_DIR  the fork's source tree
#   GIT_EXECUTABLE
##

foreach( _required PATCH_FILE SOURCE_DIR GIT_EXECUTABLE )
  if( NOT DEFINED ${_required} )
    message( FATAL_ERROR "apply_fork_patch.cmake needs -D${_required}" )
  endif()
endforeach()

if( NOT EXISTS "${PATCH_FILE}" )
  message( FATAL_ERROR "no such patch: ${PATCH_FILE}" )
endif()

execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply --reverse --check --ignore-whitespace
          "${PATCH_FILE}"
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE _already
  OUTPUT_QUIET ERROR_QUIET )

if( _already EQUAL 0 )
  message( STATUS
    "Already patched: ${SOURCE_DIR} has ${PATCH_FILE} applied" )
  return()
endif()

execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply --ignore-whitespace "${PATCH_FILE}"
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE _applied
  ERROR_VARIABLE _complaint )

if( NOT _applied EQUAL 0 )
  message( FATAL_ERROR
    "could not apply ${PATCH_FILE} to ${SOURCE_DIR}.\n"
    "Either the submodule has moved and the patch needs rebuilding, or the "
    "tree is half patched -- `git -C ${SOURCE_DIR} checkout .` resets it.\n"
    "git said:\n${_complaint}" )
endif()

message( STATUS "Patched ${SOURCE_DIR} with ${PATCH_FILE}" )
