###
# The python packages VIAME builds from source
#
# `packages/pytorch-libs/*` are research repositories with no wheel on any
# index, or with a VIAME patch applied over them. Each is built into a wheel
# and installed `--no-deps`, so what it needs comes from
# `packaging/requirements/forks.lock` and pip resolves nothing.
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
  # The condition arrives as one string, and `if( ${condition} )` would pass
  # it to if() as one argument -- a variable named "A OR B", which is never
  # set -- so every condition with more than one word was false and imgaug,
  # mmdeploy and darknet-to-pytorch-onnx were never built. Split it into
  # words first.
  separate_arguments( _viame_fork_condition UNIX_COMMAND "${condition}" )
  if( ${_viame_fork_condition} )
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
_viame_fork( sleap-nn    "VIAME_ENABLE_PYTORCH-SLEAP"       "${_pl}/sleap-nn" )
_viame_fork( roi-align   "VIAME_ENABLE_PYTORCH-MDNET"
             "${VIAME_SOURCE_DIR}/library/object_trackers/mdnet/mdnet" )

###
# The prebuilt onnxruntime C++ libraries
##
# `mmdeploy` is the only thing that wants them: it configures with
# `MMDEPLOY_TARGET_BACKENDS=ort` and needs an onnxruntime to build its ORT
# backend against. Upstream's 1.12.1 release archive, deliberately older
# than the `onnxruntime-gpu` wheel the accelerator lock pins -- the wheel is
# what runs inference, this is a headers-and-.so set mmdeploy's export
# tooling compiles against.
#
# It was `add_project_onnx.cmake`, an `ExternalProject_Add` whose whole body
# was a URL and a copy.
if( "mmdeploy" IN_LIST _viame_forks )
  if( UNIX )
    set( _ort_url "https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-linux-x64-1.12.1.tgz" )
    set( _ort_md5 "31f1cc5d934682459aaa2abb2b7ebc0f" )
  elseif( WIN32 )
    set( _ort_url "https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-win-x64-1.12.1.zip" )
    # Not pinned: the reference machine is Linux and a checksum nobody has
    # verified is worse than none, because it looks like it was.
    set( _ort_md5 "" )
  else()
    message( FATAL_ERROR
      "mmdeploy needs the onnxruntime C++ libraries and there is no build of "
      "them for this platform" )
  endif()

  set( _ort_dir
    "${VIAME_BUILD_INSTALL_PREFIX}/${python_site_packages}/onnxruntime/onnxruntimelibs" )
  set( _ort_stamp "${_viame_forks_dir}/onnxruntimelibs.stamp" )

  add_custom_command(
    OUTPUT  "${_ort_stamp}"
    COMMAND "${CMAKE_COMMAND}"
            -DURL=${_ort_url}
            -DEXPECTED_MD5=${_ort_md5}
            -DDESTINATION=${_ort_dir}
            -DDOWNLOAD_DIR=${_viame_forks_dir}/onnxruntimelibs-download
            -P "${VIAME_CMAKE_DIR}/viame_fetch_archive.cmake"
    COMMAND "${CMAKE_COMMAND}" -E touch "${_ort_stamp}"
    COMMENT "Fetching the onnxruntime C++ libraries for mmdeploy"
    VERBATIM
    )
else()
  set( _ort_stamp )
endif()

###
# One build-and-install per fork
##
# `custom_build_python_dep.cmake` is reused rather than reimplemented: it
# already skips the build when the source's git hash and the torch version
# are unchanged, which for mmcv is the difference between twenty minutes and
# nothing.
set( _viame_fork_stamps )

# What the forks with compiled code need to compile it, which the superbuild's
# add_project_pytorch.cmake set for every one of them. Without it mmcv builds
# as the pure-python `mmcv` wheel -- MMCV_WITH_OPS defaults to 0 -- and
# netharn's training fails on `mmcv._ext`; and sam2 skips its CUDA extension
# in silence, since SAM2_BUILD_ALLOW_ERRORS defaults to 1. FORCE_CUDA and
# TORCH_CUDA_ARCH_LIST because a build may have no GPU to ask (`docker build`
# has none); -allow-unsupported-compiler because Ubuntu 24.04's GCC 13.3 is
# newer than CUDA 12.6 declares support for.
set( _viame_fork_env "PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}" )
if( VIAME_ENABLE_CUDA )
  string( REPLACE ";" " " _viame_fork_arch_list "${CUDA_ARCHITECTURES}" )
  list( APPEND _viame_fork_env
    "FORCE_CUDA=1"
    "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}"
    "TORCH_CUDA_ARCH_LIST=${_viame_fork_arch_list}"
    "NVCC_APPEND_FLAGS=-allow-unsupported-compiler"
    "MMCV_CUDA_ARGS=-allow-unsupported-compiler" )
endif()
set( _viame_fork_env_mmcv "MMCV_WITH_OPS=1" )
if( VIAME_ENABLE_CUDA )
  # Stop on a failed extension build rather than install sam2 without it
  set( _viame_fork_env_sam2 "SAM2_BUILD_CUDA=1" "SAM2_BUILD_ALLOW_ERRORS=0" )
else()
  set( _viame_fork_env_sam2 "SAM2_BUILD_CUDA=0" )
endif()

set( _viame_fork_extra_deps_mmdeploy ${_ort_stamp} )

foreach( _fork IN LISTS _viame_forks )
  set( _source "${_viame_fork_source_${_fork}}" )
  set( _stamp "${_viame_forks_dir}/${_fork}.stamp" )
  set( _wheels "${_viame_forks_dir}/${_fork}" )

  set( _env ${_viame_fork_env} ${_viame_fork_env_${_fork}} )
  string( REPLACE ";" "----" _env "${_env}" )

  # A fork VIAME patches gets the patch copied over its source first. The
  # patch directories are `packages/patches/<fork>`.
  #
  # sam2's patch is for Windows, and the superbuild applied it only there. It
  # passes CUDAExtension `library_dirs=None` when it has found no Windows
  # python library directory, which torch then adds a list to, so on Linux
  # sam2 does not get past preparing its metadata.
  set( _patch_cmd "" )
  set( _viame_fork_patch_win32_only sam2 )
  if( IS_DIRECTORY "${VIAME_PATCHES_DIR}/${_fork}" AND
      ( WIN32 OR NOT _fork IN_LIST _viame_fork_patch_win32_only ) )
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
            "-DENV_VARS:STRING=${_env}"
            -DPIP_INSTALL_SCRIPT=${VIAME_CMAKE_DIR}/pip_install_with_lock.cmake
            # Install the wheel --no-deps as well as building it that way.
            # Without this a fork's first install resolved its requirements
            # from the index, around the locks and their exclusions: mmdeploy
            # brought opencv-python, a second cv2 that needs libGL, and yolo
            # brought triton and wandb. What a fork needs is in forks.lock.
            -DNO_DEPS=TRUE
            -P "${VIAME_CMAKE_DIR}/custom_build_python_dep.cmake"
    COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
    DEPENDS ${_viame_fork_extra_deps_${_fork}}
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
