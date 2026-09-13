###
# The python packages VIAME builds from source
#
# `packages/pytorch-libs/*` are research repositories with no wheel on any
# index, or with a VIAME patch applied over them. Each is built into a wheel
# and installed `--no-deps`, so what it needs comes from
# `python/requirements/forks.lock` and pip resolves nothing.
#
# `--no-deps` is the change. The superbuild installed each fork's wheel with
# dependencies, so pip fetched whatever the fork's metadata asked for at
# whatever version it felt like -- which is how `opencv-python` arrived
# beside VIAME's own cv2 (mmengine requires it), and how two installs of the
# same commit came to differ.
#
# Two of the forks are not built any more, and neither is a loss:
#
#   `packages/python-utils/pyav`  its patched `setup.py` exists to build PyAV
#                                 against a specific FFmpeg directory, which
#                                 was fletch's. P4 moved VIAME to the `av`
#                                 wheel, which carries its own FFmpeg, and
#                                 that is what made turning FFmpeg off in
#                                 fletch leave a working video stack.
#   `torchvision`                 carries no VIAME patch, and the torch index
#                                 the accelerator lock names publishes a
#                                 matching `+cuXXX` build of it.
##

if( NOT VIAME_ENABLE_PYTHON OR NOT VIAME_ENABLE_PYTORCH )
  return()
endif()

set( _viame_forks_dir "${VIAME_BINARY_DIR}/python-forks" )
set( _viame_forks )

###
# Which forks this configuration needs
##
macro( _viame_fork name condition location )
  if( ${condition} )
    # A submodule that was never checked out is an empty directory, and pip
    # given an empty directory says "neither 'setup.py' nor 'pyproject.toml'
    # found", which sends the reader looking at the wrong thing. Say what is
    # actually wrong.
    file( GLOB _viame_fork_contents "${location}/*" )

    if( NOT _viame_fork_contents )
      # Fatal only when this build is the one responsible for the python
      # environment. With `VIAME_INSTALL_PYTHON_DEPS` off, an install built
      # elsewhere is providing the forks and an empty submodule is the
      # normal state of a checkout that never needed them.
      if( VIAME_INSTALL_PYTHON_DEPS )
        message( FATAL_ERROR
          "${name} is enabled and VIAME_INSTALL_PYTHON_DEPS is on, but "
          "${location} is empty. Its submodule has not been checked out:\n"
          "    git submodule update --init --recursive ${location}\n"
          "Or turn the option that selects it off." )
      endif()

      message( STATUS
        "Not building ${name}: ${location} is empty and "
        "VIAME_INSTALL_PYTHON_DEPS is off" )
    else()
      list( APPEND _viame_forks "${name}" )
      set( _viame_fork_source_${name} "${location}" )
    endif()
  endif()
endmacro()

set( _pl "${VIAME_PACKAGES_DIR}/pytorch-libs" )

_viame_fork( imgaug      "VIAME_ENABLE_PYTORCH-MMDET OR VIAME_ENABLE_PYTORCH-NETHARN"
             "${_pl}/imgaug" )
_viame_fork( mmcv        "VIAME_ENABLE_PYTORCH-MMDET"       "${_pl}/mmcv" )
_viame_fork( mmdetection "VIAME_ENABLE_PYTORCH-MMDET"       "${_pl}/mmdetection" )
_viame_fork( mmdeploy    "VIAME_ENABLE_ONNX AND VIAME_ENABLE_PYTORCH-MMDET"
             "${_pl}/mmdeploy" )
_viame_fork( torchvideo  "VIAME_ENABLE_PYTORCH-VIDEO"       "${_pl}/torchvideo" )
_viame_fork( mit-yolo    "VIAME_ENABLE_PYTORCH-MIT-YOLO"    "${_pl}/mit-yolo" )
_viame_fork( rf-detr     "VIAME_ENABLE_PYTORCH-RF-DETR"     "${_pl}/rf-detr" )
_viame_fork( sam2        "VIAME_ENABLE_PYTORCH-SAM2"        "${_pl}/sam2" )
_viame_fork( sam3        "VIAME_ENABLE_PYTORCH-SAM3"        "${_pl}/sam3" )
_viame_fork( foundation-stereo "VIAME_ENABLE_PYTORCH-STEREO"
             "${_pl}/foundation-stereo" )
_viame_fork( detectron2  "VIAME_ENABLE_PYTORCH-DETECTRON2"  "${_pl}/detectron2" )
_viame_fork( litdet      "VIAME_ENABLE_PYTORCH-LITDET"      "${_pl}/litdet" )
_viame_fork( dino3       "VIAME_ENABLE_PYTORCH-DINO3"       "${_pl}/dino3" )
_viame_fork( darknet-to-pytorch-onnx
             "VIAME_ENABLE_ONNX AND VIAME_ENABLE_DARKNET"
             "${_pl}/darknet-to-pytorch-onnx" )
_viame_fork( roi-align   "VIAME_ENABLE_PYTORCH-MDNET"
             "${VIAME_SOURCE_DIR}/plugins/pytorch/mdnet" )

###
# One build-and-install per fork
##
# `custom_build_python_dep.cmake` is reused rather than reimplemented: it
# already skips the build when the source's git hash and the torch version
# are unchanged, which for mmcv is the difference between twenty minutes and
# nothing.
set( _viame_fork_stamps )

foreach( _fork IN LISTS _viame_forks )
  set( _source "${_viame_fork_source_${_fork}}" )
  set( _stamp "${_viame_forks_dir}/${_fork}.stamp" )
  set( _wheels "${_viame_forks_dir}/${_fork}" )

  # A fork VIAME patches gets the patch copied over its source first. The
  # patch directories are `packages/patches/<fork>`.
  set( _patch_cmd "" )
  if( IS_DIRECTORY "${VIAME_PATCHES_DIR}/${_fork}" )
    set( _patch_cmd COMMAND "${CMAKE_COMMAND}" -E copy_directory
         "${VIAME_PATCHES_DIR}/${_fork}" "${_source}" )
  endif()

  add_custom_command(
    OUTPUT  "${_stamp}"
    ${_patch_cmd}
    COMMAND "${CMAKE_COMMAND}"
            -DLIB_NAME=${_fork}
            -DLIB_SOURCE_DIR=${_source}
            -DHASH_FILE=${_viame_forks_dir}/${_fork}.hash
            -DABI_TAG=${VIAME_PYTORCH_VERSION}
            -DWHEEL_DIR=${_wheels}
            -DPython_EXECUTABLE=${Python_EXECUTABLE}
            "-DPYTHON_BUILD_CMD=${Python_EXECUTABLE}-----m----pip----wheel------no-build-isolation------no-deps------no-cache-dir------wheel-dir----${_wheels}----${_source}"
            "-DENV_VARS=PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}"
            -DPIP_INSTALL_SCRIPT=${VIAME_CMAKE_DIR}/pip_install_with_lock.cmake
            -P "${VIAME_CMAKE_DIR}/custom_build_python_dep.cmake"
    COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
    COMMENT "Building and installing ${_fork}"
    VERBATIM
    )

  list( APPEND _viame_fork_stamps "${_stamp}" )
endforeach()

if( VIAME_INSTALL_PYTHON_DEPS )
  add_custom_target( viame_python_forks ALL DEPENDS ${_viame_fork_stamps} )
else()
  add_custom_target( viame_python_forks DEPENDS ${_viame_fork_stamps} )
endif()

# The forks need what `forks.lock` pins, so they go after the dependency
# install rather than beside it.
if( TARGET viame_python_deps )
  add_dependencies( viame_python_forks viame_python_deps )
endif()
