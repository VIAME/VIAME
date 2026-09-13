###
# VIAME's python dependencies, from a lock file
#
# What this replaces is `add_project_python_deps.cmake` plus the pip half of
# `add_project_pytorch.cmake`: about six hundred lines that computed a
# dependency list out of thirty `if()` branches, handed it to pip as loose
# requirements, and let pip resolve whatever it liked on the day. Two builds
# of the same VIAME commit a month apart did not get the same environment,
# and the difference was invisible -- there was no file to diff.
#
# `python/requirements/*.in` says what VIAME needs and why; `*.lock` is the
# pinned resolution `pip-compile` produced from it, committed. This installs
# the lock, `--no-deps`, so pip resolves nothing at build time.
##

option( VIAME_INSTALL_PYTHON_DEPS
  "Install VIAME's python dependencies as part of the build" ON )

set( VIAME_PYTHON_INDEX_URL "" CACHE STRING
  "Extra index to install python dependencies from; the lock names its own \
when it needs one (torch's CUDA builds are not on PyPI)" )

mark_as_advanced( VIAME_PYTHON_INDEX_URL )

# Three packages cannot be used as published and are edited in place after
# they are installed -- `torch.load`'s `weights_only` default and ubelt's
# removal of `ensure_unicode`. `python/patches/apply.py` carries what
# `custom_install_viame.cmake` did, and unlike it says so when a patch stops
# matching rather than doing nothing. P9 removes the step: the patched
# packages become wheels the wheel CI builds.

# Defaulted ON, which `lite-build-system.md` section 2 asks for, after the
# run that had to happen first. Against an install built the old way it
# moves 145 package versions, including replacing a torchvision built from
# source with the matching `+cu126` build from the index the lock names.
# Everything passes on the result: 426 unit and baseline tests, 2 golden, 7
# critical. A from-scratch build into an empty prefix, whose python came
# only from these locks, passes 409 of 409 unit and core tests.
#
# A build with this on is responsible for its own python environment, which
# includes the forks in `packages/pytorch-libs`, so `viame_python_forks.cmake`
# asks for those submodules and says so if they are not checked out.

if( NOT VIAME_ENABLE_PYTHON )
  return()
endif()

set( _viame_req_dir "${VIAME_SOURCE_DIR}/python/requirements" )

###
# Which locks this configuration needs
##
if( VIAME_ENABLE_CUDA )
  if( CUDA_VERSION VERSION_GREATER_EQUAL "13.0" )
    set( _viame_locks "${_viame_req_dir}/cuda13.lock" )
  else()
    set( _viame_locks "${_viame_req_dir}/cuda12.lock" )
  endif()
else()
  set( _viame_locks "${_viame_req_dir}/cpu.lock" )
endif()

# The forks in `packages/pytorch-libs` are built from source and installed
# `--no-deps`, so what they need has to be asked for here. See `forks.in`.
if( VIAME_ENABLE_PYTORCH )
  list( APPEND _viame_locks "${_viame_req_dir}/forks.lock" )
endif()

if( VIAME_ENABLE_PYTORCH-LEARN )
  list( APPEND _viame_locks "${_viame_req_dir}/learn.lock" )
endif()

if( VIAME_ENABLE_PYTORCH-SLEAP )
  list( APPEND _viame_locks "${_viame_req_dir}/sleap.lock" )
endif()

if( VIAME_ENABLE_COLMAP )
  list( APPEND _viame_locks "${_viame_req_dir}/colmap.lock" )
endif()

if( VIAME_ENABLE_TESTS )
  list( APPEND _viame_locks "${_viame_req_dir}/test.lock" )
endif()

###
# The install
##
set( _viame_pip_args )
foreach( _lock IN LISTS _viame_locks )
  list( APPEND _viame_pip_args "-r" "${_lock}" )
endforeach()

if( VIAME_PYTHON_INDEX_URL )
  list( APPEND _viame_pip_args "--extra-index-url" "${VIAME_PYTHON_INDEX_URL}" )
endif()

set( _viame_deps_stamp
  "${CMAKE_CURRENT_BINARY_DIR}/viame_python_deps.stamp" )

# `DEPENDS` on the locks rather than a hash of them: a lock file is the
# input, so make already knows how to decide whether this is out of date.
#
# `--no-deps` is the point of the exercise -- everything the install needs
# is named in a lock, so pip is a fetcher here and not a resolver. It is
# also what keeps `opencv-python` and `wandb` out: two of the forks require
# them, and a resolving install would put a second cv2 beside VIAME's.
add_custom_command(
  OUTPUT  "${_viame_deps_stamp}"
  COMMAND "${CMAKE_COMMAND}" -E env
            "PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}"
            "PYTHONNOUSERSITE="
          "${Python_EXECUTABLE}" -m pip install --user --no-deps
            --no-warn-script-location ${_viame_pip_args}
  COMMAND "${Python_EXECUTABLE}"
          "${VIAME_SOURCE_DIR}/python/patches/apply.py"
          --site-packages
            "${VIAME_BUILD_INSTALL_PREFIX}/${python_site_packages}"
  COMMAND "${CMAKE_COMMAND}" -E touch "${_viame_deps_stamp}"
  DEPENDS ${_viame_locks}
          "${VIAME_SOURCE_DIR}/python/patches/apply.py"
  COMMENT "Installing VIAME's python dependencies from the lock files"
  VERBATIM
  )

if( VIAME_INSTALL_PYTHON_DEPS )
  add_custom_target( viame_python_deps ALL DEPENDS "${_viame_deps_stamp}" )
else()
  add_custom_target( viame_python_deps DEPENDS "${_viame_deps_stamp}" )
endif()
