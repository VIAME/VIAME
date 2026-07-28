# custom_fetch_git_package.cmake
#
# Clone, or re-sync, a source package that is fetched on demand instead of
# being carried as a superbuild git submodule. OnDemandGitPackage() in
# common_macros.cmake generates the invocation; this is what runs at build
# time, via cmake -P.
#
# Required variables:
#   PKG_NAME   - Package name, used in status messages
#   REPO_URL   - Git remote to clone from
#   TARGET_DIR - Directory the checkout lives in
#
# Optional variables:
#   GIT_REF    - Tag, branch or full sha to check out (default: default branch)
#   RECURSIVE  - Also init/update the package's own submodules
#   SHALLOW    - Clone and fetch with --depth 1
#   TRACK      - Fetch on every run rather than only when the checkout no
#                longer matches GIT_REF. Needed for refs that move (branches),
#                pointless for tags and shas.

cmake_minimum_required( VERSION 3.16 )

if( NOT PKG_NAME OR NOT REPO_URL OR NOT TARGET_DIR )
  message( FATAL_ERROR
    "custom_fetch_git_package.cmake requires PKG_NAME, REPO_URL and TARGET_DIR" )
endif()

find_package( Git QUIET )

if( NOT GIT_EXECUTABLE )
  set( GIT_EXECUTABLE git )
endif()

# Run a git command, aborting the build if it fails
function( RunGit _description )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} ${ARGN}
    RESULT_VARIABLE _result )
  if( NOT _result EQUAL 0 )
    message( FATAL_ERROR "${PKG_NAME}: ${_description} failed with code ${_result}" )
  endif()
endfunction()

# Run a git command, returning its trimmed output, or "" if it failed
function( GitQuery _output_var )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} ${ARGN}
    OUTPUT_VARIABLE _stdout
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _result )
  if( NOT _result EQUAL 0 )
    set( _stdout "" )
  endif()
  set( ${_output_var} "${_stdout}" PARENT_SCOPE )
endfunction()

# Resolve a ref to a commit within the checkout. The remote-tracking copy wins
# so that a branch follows the last fetch rather than a stale local branch;
# tags and shas have no remote-tracking form and fall through to the ref itself.
function( ResolveRef _ref _output_var )
  set( _sha "" )
  if( _ref )
    GitQuery( _sha -C "${TARGET_DIR}" rev-parse --verify --quiet
      "refs/remotes/origin/${_ref}^{commit}" )
    if( NOT _sha )
      GitQuery( _sha -C "${TARGET_DIR}" rev-parse --verify --quiet "${_ref}^{commit}" )
    endif()
  endif()
  set( ${_output_var} "${_sha}" PARENT_SCOPE )
endfunction()

# A raw sha cannot be handed to `git clone --branch`, and servers commonly
# refuse a shallow fetch of an arbitrary sha, so pin-by-sha uses a full clone
set( REF_IS_SHA FALSE )
if( GIT_REF MATCHES "^[0-9a-fA-F]{40}$" )
  set( REF_IS_SHA TRUE )
  set( SHALLOW FALSE )
endif()

# A leftover submodule checkout has a .git file rather than a directory, so ask
# git rather than testing for a directory
set( HAVE_CHECKOUT FALSE )
if( EXISTS "${TARGET_DIR}/.git" )
  GitQuery( _git_dir -C "${TARGET_DIR}" rev-parse --git-dir )
  if( _git_dir )
    set( HAVE_CHECKOUT TRUE )
  endif()
endif()

if( NOT HAVE_CHECKOUT )
  if( EXISTS "${TARGET_DIR}" )
    file( GLOB _leftovers "${TARGET_DIR}/*" "${TARGET_DIR}/.??*" )
    if( _leftovers )
      message( STATUS "${PKG_NAME}: removing non-git directory ${TARGET_DIR}" )
      file( REMOVE_RECURSE "${TARGET_DIR}" )
    endif()
  endif()

  get_filename_component( _parent_dir "${TARGET_DIR}" DIRECTORY )
  file( MAKE_DIRECTORY "${_parent_dir}" )

  set( _clone_args clone )
  if( SHALLOW )
    list( APPEND _clone_args --depth 1 )
  endif()
  if( GIT_REF AND NOT REF_IS_SHA )
    list( APPEND _clone_args --branch "${GIT_REF}" )
  endif()
  list( APPEND _clone_args "${REPO_URL}" "${TARGET_DIR}" )

  message( STATUS "${PKG_NAME}: cloning ${REPO_URL} into ${TARGET_DIR}" )
  RunGit( "clone of ${REPO_URL}" ${_clone_args} )
endif()

# Keep the remote pointed at the configured url, so a moved upstream is picked
# up without anyone having to delete the checkout by hand
GitQuery( _origin_url -C "${TARGET_DIR}" remote get-url origin )

if( NOT "${_origin_url}" STREQUAL "${REPO_URL}" )
  if( _origin_url )
    message( STATUS "${PKG_NAME}: origin url ${_origin_url} -> ${REPO_URL}" )
    RunGit( "remote set-url" -C "${TARGET_DIR}" remote set-url origin "${REPO_URL}" )
  else()
    RunGit( "remote add" -C "${TARGET_DIR}" remote add origin "${REPO_URL}" )
  endif()
endif()

GitQuery( _head_sha -C "${TARGET_DIR}" rev-parse --verify --quiet "HEAD^{commit}" )
ResolveRef( "${GIT_REF}" _want_sha )

# Nothing to do, and no network round trip, while the pin still matches what is
# checked out. A bumped pin no longer resolves to HEAD and drops through
set( NEED_FETCH TRUE )
if( GIT_REF AND NOT TRACK AND _want_sha AND _want_sha STREQUAL "${_head_sha}" )
  set( NEED_FETCH FALSE )
endif()

if( NEED_FETCH )
  set( _fetch_args -C "${TARGET_DIR}" fetch --force --tags )
  if( SHALLOW )
    list( APPEND _fetch_args --depth 1 )
  endif()
  list( APPEND _fetch_args origin )
  if( GIT_REF )
    list( APPEND _fetch_args "${GIT_REF}" )
  endif()

  message( STATUS "${PKG_NAME}: fetching ${GIT_REF} from ${REPO_URL}" )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} ${_fetch_args}
    RESULT_VARIABLE _fetch_result )

  if( NOT _fetch_result EQUAL 0 )
    # Not every server serves a ref asked for by name, notably raw shas, so try
    # everything the remote has before giving up
    message( STATUS "${PKG_NAME}: fetch of ${GIT_REF} failed, fetching all refs" )
    RunGit( "fetch from ${REPO_URL}" -C "${TARGET_DIR}" fetch --force --tags origin )
  endif()

  ResolveRef( "${GIT_REF}" _want_sha )
  if( NOT _want_sha )
    GitQuery( _want_sha -C "${TARGET_DIR}" rev-parse --verify --quiet "FETCH_HEAD^{commit}" )
  endif()
  if( NOT _want_sha )
    message( FATAL_ERROR "${PKG_NAME}: unable to resolve '${GIT_REF}' in ${TARGET_DIR}" )
  endif()
endif()

if( NOT _want_sha STREQUAL "${_head_sha}" )
  # --force because build steps patch these trees in place; the patch step runs
  # after this one and puts its overlay back
  message( STATUS "${PKG_NAME}: checking out ${GIT_REF} (${_want_sha})" )
  RunGit( "checkout of ${GIT_REF}" -C "${TARGET_DIR}"
    checkout --force --detach "${_want_sha}" )
else()
  message( STATUS "${PKG_NAME}: already at ${GIT_REF} (${_head_sha})" )
endif()

if( CMAKE_HOST_WIN32 )
  # Some packages, pytorch's composable_kernel among them, carry submodule
  # paths past the 260 character default limit and fail the update without this
  execute_process(
    COMMAND ${GIT_EXECUTABLE} -C "${TARGET_DIR}" config core.longpaths true )
endif()

if( RECURSIVE )
  message( STATUS "${PKG_NAME}: updating submodules" )
  RunGit( "submodule update" -C "${TARGET_DIR}" submodule update --init --recursive )
endif()
