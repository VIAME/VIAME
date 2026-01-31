# PyTorch External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

CreateDirectory( ${VIAME_BUILD_PREFIX}/src/pytorch-build )

set( PYTORCH_LIBS_TO_BUILD )

if( VIAME_BUILD_PYTORCH_FROM_SOURCE )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} pytorch )
endif()

if( VIAME_BUILD_TORCHVISION_FROM_SOURCE )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} torchvision )
endif()

if( VIAME_PYTHON_DEPS_REQ_TORCH )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} pytorch-libs-deps )
endif()

if( VIAME_ENABLE_PYTORCH-VIDEO )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} pyav torchvideo )
endif()

if( VIAME_ENABLE_PYTORCH-MMDET OR VIAME_ENABLE_PYTORCH-NETHARN )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} imgaug )
endif()

if( VIAME_ENABLE_PYTORCH-MMDET )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} mmcv mmdetection )
endif()

if( VIAME_ENABLE_ONNX AND VIAME_ENABLE_DARKNET)
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} darknet-to-pytorch-onnx )
endif()

if( VIAME_ENABLE_PYTORCH-MIT-YOLO )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} mit-yolo )
endif()

if( VIAME_ENABLE_ONNX AND VIAME_ENABLE_PYTORCH-MMDET )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} mmdeploy )
endif()

if( VIAME_ENABLE_PYTORCH-MDNET )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} roi-align )
endif()

if( VIAME_ENABLE_PYTORCH-DETECTRON2 )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} detectron2 )
endif()

if( VIAME_ENABLE_PYTORCH-SAM2 )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} sam2 )
endif()

if( VIAME_ENABLE_PYTORCH-SAM3 )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} sam3 )
endif()

if( VIAME_ENABLE_PYTORCH-STEREO )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} foundation-stereo )
endif()

if( VIAME_ENABLE_PYTORCH-RF-DETR )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} rf-detr )
endif()

if( VIAME_ENABLE_PYTORCH-LITDET )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} litdet )
endif()

if( VIAME_ENABLE_TENSORRT )
  set( PYTORCH_LIBS_TO_BUILD ${PYTORCH_LIBS_TO_BUILD} torch2rt )
endif()

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} ${PYTORCH_LIBS_TO_BUILD} )
set( PYTORCH_ENV_VARS ${PYTHON_DEP_ENV_VARS} )

if( VIAME_ENABLE_CUDNN )
  if( WIN32 )
    string( REPLACE ";" "----" CUDNN_ADJ_LIB_LIST "${CUDNN_LIBRARIES}" )
    list( APPEND PYTORCH_ENV_VARS "CUDNN_LIBRARY=${CUDNN_ADJ_LIB_LIST}" )
  else()
    list( APPEND PYTORCH_ENV_VARS "CUDNN_LIBRARY=${CUDNN_LIBRARIES}" )
  endif()
  if( CUDNN_ROOT_DIR )
    list( APPEND PYTORCH_ENV_VARS "CUDNN_INCLUDE_DIR=${CUDNN_ROOT_DIR}/include" )
  endif()
endif()

if( VIAME_BUILD_LIMIT_NINJA )
  list( APPEND PYTORCH_ENV_VARS "USE_NINJA=OFF" )
endif()

if( VIAME_ENABLE_PYTORCH-NETHARN )
  list( APPEND PYTORCH_ENV_VARS "BEZIER_NO_EXTENSION=1" )
endif()

if( VIAME_ENABLE_CUDA )
  list( APPEND PYTORCH_ENV_VARS "USE_CUDA=1" )
  list( APPEND PYTORCH_ENV_VARS "FORCE_CUDA=1" )
  list( APPEND PYTORCH_ENV_VARS "CUDA_VISIBLE_DEVICES=0" )
  list( APPEND PYTORCH_ENV_VARS "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}" )
  list( APPEND PYTORCH_ENV_VARS "CUDACXX=${CUDA_NVCC_EXECUTABLE}" )
  list( APPEND PYTORCH_ENV_VARS "TORCH_CUDA_ARCH_LIST=${CUDA_ARCHITECTURES}" )
  list( APPEND PYTORCH_ENV_VARS "TORCH_NVCC_FLAGS=-Xfatbin -compress-all" )
  list( APPEND PYTORCH_ENV_VARS "NVCC_FLAGS=-allow-unsupported-compiler" )
  list( APPEND PYTORCH_ENV_VARS "CUDAFLAGS=-allow-unsupported-compiler" )
  list( APPEND PYTORCH_ENV_VARS "CMAKE_CUDA_FLAGS=--allow-unsupported-compiler" )
  list( APPEND PYTORCH_ENV_VARS "MMCV_CUDA_ARGS=-allow-unsupported-compiler" )
  list( APPEND PYTORCH_ENV_VARS "NO_CAFFE2_OPS=1" )
else()
  list( APPEND PYTORCH_ENV_VARS "USE_CUDA=0" )
endif()

if( VIAME_ENABLE_PYTORCH-MMDET )
  list( APPEND PYTORCH_ENV_VARS "MMCV_WITH_OPS=1" )
endif()

if( VIAME_BUILD_TORCHVISION_FROM_SOURCE AND NOT WIN32 )
  list( APPEND PYTORCH_ENV_VARS "TORCHVISION_USE_PNG=0" )
endif()

if( WIN32 AND VIAME_ENABLE_PYTORCH-LEARN )
  list( APPEND PYTORCH_ENV_VARS "SETUPTOOLS_USE_DISTUTILS=1" )
endif()

if( WIN32 AND VIAME_BUILD_PYTORCH_FROM_SOURCE )
  list( APPEND PYTORCH_ENV_VARS "DISTUTILS_USE_SDK=1" )
  list( APPEND PYTORCH_ENV_VARS "CMAKE_PREFIX_PATH=${VIAME_INSTALL_PREFIX}" )
  list( APPEND PYTORCH_ENV_VARS "USE_DISTRIBUTED=0" )
  list( APPEND PYTORCH_ENV_VARS "USE_NCCL=0" )
  list( APPEND PYTORCH_ENV_VARS "CC=${CMAKE_C_COMPILER}" )
  list( APPEND PYTORCH_ENV_VARS "CXX=${CMAKE_CXX_COMPILER}" )

  # Prepend pytorch source dir to PYTHONPATH so PyTorch's own 'tools' package
  # takes priority over any conflicting package (e.g. detectron2 installs a
  # 'tools' package into site-packages which shadows PyTorch's tools.pyi)
  list( FILTER PYTORCH_ENV_VARS EXCLUDE REGEX "^PYTHONPATH=" )
  list( APPEND PYTORCH_ENV_VARS
    "PYTHONPATH=${VIAME_PACKAGES_DIR}/pytorch<PS>${VIAME_PYTHON_PATH}" )
endif()

# On Windows, add torch lib directory to PATH so DLLs can be found when importing torch
# This is required because Python 3.8+ changed DLL search behavior on Windows
if( WIN32 )
  set( TORCH_DLL_PATH "${VIAME_PYTHON_PACKAGES}/torch/lib<PS>${VIAME_INSTALL_PREFIX}/bin" )
  if( VIAME_ENABLE_CUDA AND CUDA_TOOLKIT_ROOT_DIR )
    set( TORCH_DLL_PATH "${TORCH_DLL_PATH}<PS>${CUDA_TOOLKIT_ROOT_DIR}/bin" )
  endif()

  # For torchvision and other PyTorch extensions on Windows, we need to provide
  # MSVC and Windows SDK include/lib paths since newer VS versions (2026+) may not
  # have them automatically available when Python's setuptools runs the compiler.
  if( CMAKE_CXX_COMPILER_ID MATCHES MSVC )
    # Get MSVC include directory from compiler path
    # Compiler is at: .../VC/Tools/MSVC/version/bin/Hostx64/x64/cl.exe
    # Includes are at: .../VC/Tools/MSVC/version/include
    get_filename_component( MSVC_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY )
    get_filename_component( MSVC_INCLUDE_DIR "${MSVC_BIN_DIR}/../../../include" ABSOLUTE )
    get_filename_component( MSVC_LIB_DIR "${MSVC_BIN_DIR}/../../../lib/x64" ABSOLUTE )

    # Add MSVC bin directory to PATH so cl.exe is discoverable by pytorch's CMake
    set( TORCH_DLL_PATH "${TORCH_DLL_PATH}<PS>${MSVC_BIN_DIR}" )

    # Find Windows SDK path
    set( WIN_SDK_ROOT "C:/Program Files (x86)/Windows Kits/10" )
    if( EXISTS "${WIN_SDK_ROOT}/Include" )
      file( GLOB SDK_VERSIONS "${WIN_SDK_ROOT}/Include/10.*" )
      if( SDK_VERSIONS )
        list( SORT SDK_VERSIONS )
        list( GET SDK_VERSIONS -1 SDK_VERSION_PATH )
        get_filename_component( SDK_VERSION "${SDK_VERSION_PATH}" NAME )

        set( SDK_UCRT_INCLUDE "${WIN_SDK_ROOT}/Include/${SDK_VERSION}/ucrt" )
        set( SDK_SHARED_INCLUDE "${WIN_SDK_ROOT}/Include/${SDK_VERSION}/shared" )
        set( SDK_UM_INCLUDE "${WIN_SDK_ROOT}/Include/${SDK_VERSION}/um" )
        set( SDK_UCRT_LIB "${WIN_SDK_ROOT}/Lib/${SDK_VERSION}/ucrt/x64" )
        set( SDK_UM_LIB "${WIN_SDK_ROOT}/Lib/${SDK_VERSION}/um/x64" )

        # Add Windows SDK bin directory to PATH for rc.exe (Resource Compiler)
        # The linker needs rc.exe which is in the SDK bin directory
        set( SDK_BIN_X64 "${WIN_SDK_ROOT}/bin/${SDK_VERSION}/x64" )
        if( EXISTS "${SDK_BIN_X64}/rc.exe" )
          set( TORCH_DLL_PATH "${TORCH_DLL_PATH}<PS>${SDK_BIN_X64}" )
          message( STATUS "VIAME: Added Windows SDK bin to PATH: ${SDK_BIN_X64}" )
        endif()

        # Merge MSVC/SDK paths with existing INCLUDE/LIB from PYTHON_DEP_ENV_VARS
        # Remove existing entries first, then add combined versions so both
        # MSVC/SDK paths and VIAME install/system paths are preserved
        set( MSVC_INCLUDE_PATHS "${MSVC_INCLUDE_DIR}<PS>${SDK_UCRT_INCLUDE}<PS>${SDK_SHARED_INCLUDE}<PS>${SDK_UM_INCLUDE}" )
        set( MSVC_LIB_PATHS "${MSVC_LIB_DIR}<PS>${SDK_UCRT_LIB}<PS>${SDK_UM_LIB}" )

        list( FILTER PYTORCH_ENV_VARS EXCLUDE REGEX "^INCLUDE=" )
        list( FILTER PYTORCH_ENV_VARS EXCLUDE REGEX "^LIB=" )
        list( APPEND PYTORCH_ENV_VARS "INCLUDE=${MSVC_INCLUDE_PATHS}<PS>${ADJ_INCLUDE_PATH}" )
        list( APPEND PYTORCH_ENV_VARS "LIB=${MSVC_LIB_PATHS}<PS>${ADJ_LIBRARY_PATH}" )

        message( STATUS "VIAME: Set INCLUDE/LIB for MSVC ${CMAKE_CXX_COMPILER_VERSION} and Windows SDK ${SDK_VERSION}" )
      endif()
    endif()
  endif()

  # Prepend our paths to existing PATH (use <PS> as placeholder for path separator)
  # $ENV{PATH} captures the PATH at configure time, which should include essential system paths
  # Must escape semicolons in $ENV{PATH} by converting them to <PS> placeholders,
  # otherwise they get treated as CMake list separators and break the command
  string( REPLACE ";" "<PS>" ESCAPED_ENV_PATH "$ENV{PATH}" )
  list( APPEND PYTORCH_ENV_VARS "PATH=${TORCH_DLL_PATH}<PS>${ESCAPED_ENV_PATH}" )
endif()

foreach( LIB ${PYTORCH_LIBS_TO_BUILD} )
  if( "${LIB}" STREQUAL "pytorch" )
    set( LIBRARY_LOCATION ${VIAME_PACKAGES_DIR}/pytorch )
  elseif( "${LIB}" STREQUAL "pytorch-libs-deps" )
    set( LIBRARY_LOCATION ${VIAME_CMAKE_DIR} )
  elseif( "${LIB}" STREQUAL "roi-align" )
    set( LIBRARY_LOCATION ${VIAME_SOURCE_DIR}/plugins/pytorch/mdnet )
  elseif( "${LIB}" STREQUAL "pyav" )
    set( LIBRARY_LOCATION ${VIAME_PACKAGES_DIR}/python-utils/pyav )
  else()
    set( LIBRARY_LOCATION ${VIAME_PACKAGES_DIR}/pytorch-libs/${LIB} )
  endif()

  set( LIBRARY_PIP_BUILD_DIR ${VIAME_BUILD_PREFIX}/src/pytorch-build/${LIB}-build )
  CreateDirectory( ${LIBRARY_PIP_BUILD_DIR} )
  set( USE_BUILD_SCRIPT_FOR_INSTALL FALSE )

  if( "${LIB}" STREQUAL "pytorch-libs-deps" )
    # pytorch-libs-deps is a pip install of torch-dependent packages, not a source build
    set( LIBRARY_PIP_BUILD_CMD "" )
    string( REPLACE ";" " " TORCH_DEPS_STR "${VIAME_PYTHON_DEPS_REQ_TORCH}" )
    set( PIP_CMD "pip install --user" )
    if( VIAME_BUILD_NO_CACHE_DIR )
      set( PIP_CMD "${PIP_CMD} --no-cache-dir" )
    endif()
    set( PIP_CMD "${PIP_CMD} ${TORCH_DEPS_STR}" )
    set( PIP_CMD "${PIP_CMD} --no-deps" )
    string( REPLACE " " ";" PIP_CMD "${PIP_CMD}" )
    set( LIBRARY_PIP_INSTALL_CMD ${Python_EXECUTABLE} -m ${PIP_CMD} )
  elseif( VIAME_PYTHON_SYMLINK )
    if( "${LIB}" STREQUAL "mit-yolo" OR "${LIB}" STREQUAL "rf-detr" OR "${LIB}" STREQUAL "litdet" OR "${LIB}" STREQUAL "sam3" )
      set( LIBRARY_PIP_BUILD_CMD "" )
      if( VIAME_BUILD_NO_CACHE_DIR )
        set( LIBRARY_PIP_INSTALL_CMD
          ${Python_EXECUTABLE} -m pip install --no-build-isolation --no-cache-dir --user -e . )
      else()
        set( LIBRARY_PIP_INSTALL_CMD
          ${Python_EXECUTABLE} -m pip install --no-build-isolation --user -e . )
      endif()
    else()
      set( LIBRARY_PIP_BUILD_CMD
        ${Python_EXECUTABLE} setup.py build --build-base=${LIBRARY_PIP_BUILD_DIR} )
      if( VIAME_BUILD_NO_CACHE_DIR )
        set( LIBRARY_PIP_INSTALL_CMD
          ${Python_EXECUTABLE} -m pip install --no-cache-dir --user -e . )
      else()
        set( LIBRARY_PIP_INSTALL_CMD
          ${Python_EXECUTABLE} -m pip install --user -e . )
      endif()
    endif()
  else()
    if( "${LIB}" STREQUAL "mit-yolo" OR "${LIB}" STREQUAL "rf-detr" OR "${LIB}" STREQUAL "litdet" OR "${LIB}" STREQUAL "sam3" )
      # Use pip wheel for pyproject.toml-based packages
      # This avoids creating build directories in source tree
      # Must use --no-cache-dir to ensure wheel is written to --wheel-dir (not just cached)
      set( LIBRARY_PIP_BUILD_CMD
        ${Python_EXECUTABLE} -m pip wheel
          --no-build-isolation
          --no-deps
          --no-cache-dir
          --wheel-dir ${LIBRARY_PIP_BUILD_DIR}
          ${LIBRARY_LOCATION}
      )
    elseif( "${LIB}" STREQUAL "pytorch" OR "${LIB}" STREQUAL "mmcv" OR "${LIB}" STREQUAL "torchvision" )
      # Use pip wheel instead of setup.py bdist_wheel to avoid Windows cleanup
      # errors ("no such file or directory" when removing bdist temp directory)
      # Must use --no-cache-dir to ensure wheel is written to --wheel-dir (not just cached)
      set( LIBRARY_PIP_BUILD_CMD
        ${Python_EXECUTABLE} -m pip wheel
          --no-build-isolation
          --no-deps
          --no-cache-dir
          --wheel-dir ${LIBRARY_PIP_BUILD_DIR}
          ${LIBRARY_LOCATION}
      )
    else()
      set( LIBRARY_PIP_BUILD_CMD
        ${Python_EXECUTABLE} setup.py
          build --build-base=${LIBRARY_PIP_BUILD_DIR}
          build_ext
            --include-dirs=${VIAME_INSTALL_PREFIX}/include
            --library-dirs=${VIAME_INSTALL_PREFIX}/lib
          bdist_wheel -d ${LIBRARY_PIP_BUILD_DIR} )
    endif()
    # Wheel install is handled by the build script (custom_build_python_dep.cmake)
    # so it can pass FORCE_REINSTALL based on whether a rebuild occurred
    set( LIBRARY_PIP_INSTALL_CMD "" )
    set( USE_BUILD_SCRIPT_FOR_INSTALL TRUE )
  endif()

  # Convert install command and env vars to ----separated strings for the wrapper script
  if( USE_BUILD_SCRIPT_FOR_INSTALL )
    # Wheel install is handled by build script, use no-op for install step
    set( LIBRARY_PYTHON_INSTALL ${CMAKE_COMMAND} -E echo "Install handled by build step" )
  else()
    set( PYTORCH_INSTALL_ENV_VARS ${PYTORCH_ENV_VARS} "PYTORCH_BUILD_DIR=${LIBRARY_PIP_BUILD_DIR}" )
    string( REPLACE ";" "----" PYTORCH_INSTALL_CMD_STR "${LIBRARY_PIP_INSTALL_CMD}" )
    string( REPLACE ";" "----" PYTORCH_INSTALL_ENV_STR "${PYTORCH_INSTALL_ENV_VARS}" )

    set( LIBRARY_PYTHON_INSTALL
      ${CMAKE_COMMAND}
        -DCOMMAND_TO_RUN:STRING=${PYTORCH_INSTALL_CMD_STR}
        -DENV_VARS:STRING=${PYTORCH_INSTALL_ENV_STR}
        -DWORKING_DIR:PATH=${LIBRARY_LOCATION}
        -P ${VIAME_CMAKE_DIR}/run_python_command.cmake )
  endif()

  set( LIBRARY_PATCH_COMMAND "" )
  set( PROJECT_DEPS fletch python-deps )

  if( NOT "${LIB}" STREQUAL "pytorch" )
    set( PROJECT_DEPS ${PROJECT_DEPS} pytorch )
    if( VIAME_ENABLE_PYTORCH-VISION AND
        NOT "${LIB}" STREQUAL "torchvision" )
      set( PROJECT_DEPS ${PROJECT_DEPS} torchvision )
    endif()
  endif()

  if( "${LIB}" STREQUAL "pytorch" )
    set( LIBRARY_PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${VIAME_PATCHES_DIR}/pytorch
      ${VIAME_PACKAGES_DIR}/pytorch )
  elseif( "${LIB}" STREQUAL "torch2rt" )
    set( PROJECT_DEPS fletch python-deps tensorrt )
  elseif( "${LIB}" STREQUAL "detectron2" )
    if( WIN32 )
      set( LIBRARY_PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${VIAME_PATCHES_DIR}/detectron2
        ${VIAME_PACKAGES_DIR}/pytorch-libs/detectron2 )
    endif()
  elseif( "${LIB}" STREQUAL "pyav" )
    # On Windows, FFmpeg puts .lib files in bin/ instead of lib/
    # Need to include both directories in library search path
    # Use separate --library-dirs flags to avoid semicolon escaping issues
    # with the ---- list separator mechanism used by ExternalProject_Add
    if( WIN32 )
      set( LIBRARY_PIP_BUILD_CMD
        ${Python_EXECUTABLE} setup.py
          build --build-base=${LIBRARY_PIP_BUILD_DIR}
          build_ext
            --include-dirs=${VIAME_INSTALL_PREFIX}/include
            --library-dirs=${VIAME_INSTALL_PREFIX}/lib
            --library-dirs=${VIAME_INSTALL_PREFIX}/bin
          bdist_wheel -d ${LIBRARY_PIP_BUILD_DIR} )
    endif()
    set( LIBRARY_PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${VIAME_PATCHES_DIR}/pyav
      ${VIAME_PACKAGES_DIR}/python-utils/pyav )
  elseif( "${LIB}" STREQUAL "torchvideo" )
    set( PROJECT_DEPS ${PROJECT_DEPS} pyav )
  elseif( "${LIB}" STREQUAL "mmcv" )
    if( WIN32 )
      set( LIBRARY_PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${VIAME_PATCHES_DIR}/mmcv
        ${VIAME_PACKAGES_DIR}/pytorch-libs/mmcv )
    endif()
  elseif( "${LIB}" STREQUAL "sam2" )
    if( WIN32 )
      set( LIBRARY_PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${VIAME_PATCHES_DIR}/sam2
        ${VIAME_PACKAGES_DIR}/pytorch-libs/sam2 )
    endif()
  elseif( "${LIB}" STREQUAL "mmdetection" )
    set( PROJECT_DEPS ${PROJECT_DEPS} mmcv )
  elseif( "${LIB}" STREQUAL "mmdeploy" )
    set( PROJECT_DEPS ${PROJECT_DEPS} mmdetection onnxruntimelibs )
  elseif( "${LIB}" STREQUAL "mit-yolo" )
    set( LIBRARY_PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${VIAME_PATCHES_DIR}/mit-yolo
      ${VIAME_PACKAGES_DIR}/pytorch-libs/mit-yolo )
  elseif( "${LIB}" STREQUAL "rf-detr" )
    set( LIBRARY_PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${VIAME_PATCHES_DIR}/rf-detr
      ${VIAME_PACKAGES_DIR}/pytorch-libs/rf-detr )
  elseif( "${LIB}" STREQUAL "detectron2" )
    set( PROJECT_DEPS ${PROJECT_DEPS} pytorch-libs-deps )
  elseif( "${LIB}" STREQUAL "sam3" )
    set( LIBRARY_PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${VIAME_PATCHES_DIR}/sam3
      ${VIAME_PACKAGES_DIR}/pytorch-libs/sam3 )
    set( PROJECT_DEPS ${PROJECT_DEPS} pytorch-libs-deps )
  elseif( "${LIB}" STREQUAL "foundation-stereo" )
    set( LIBRARY_PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${VIAME_PATCHES_DIR}/foundation-stereo
      ${VIAME_PACKAGES_DIR}/pytorch-libs/foundation-stereo )
    set( PROJECT_DEPS ${PROJECT_DEPS} pytorch-libs-deps )
  elseif( "${LIB}" STREQUAL "darknet-to-pytorch-onnx" )
    set( LIBRARY_PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${VIAME_PATCHES_DIR}/darknet-to-pytorch-onnx
      ${VIAME_PACKAGES_DIR}/pytorch-libs/darknet-to-pytorch-onnx )
  endif()

  # Use conditional build that checks source hash
  # This prevents unnecessary recompilation when source hasn't changed
  set( LIB_HASH_FILE ${VIAME_BUILD_PREFIX}/src/${LIB}-source-hash.txt )

  # Convert lists to ----separated strings for passing through ExternalProject_Add
  set( PYTORCH_ENV_VARS_WITH_BUILD_DIR ${PYTORCH_ENV_VARS} "PYTORCH_BUILD_DIR=${LIBRARY_PIP_BUILD_DIR}" )

  # Set SAM2_BUILD_CUDA based on whether CUDA is enabled
  if( "${LIB}" STREQUAL "sam2" )
    if( VIAME_ENABLE_CUDA )
      list( APPEND PYTORCH_ENV_VARS_WITH_BUILD_DIR "SAM2_BUILD_CUDA=1" )
      # On Windows, set CL environment variable to fix 'std' ambiguous symbol error
      # in PyTorch headers (compiled_autograd.h, tree_views.h) when compiling CUDA extensions
      # See: https://github.com/pytorch/pytorch/issues/166123
      if( WIN32 )
        list( APPEND PYTORCH_ENV_VARS_WITH_BUILD_DIR "CL=/permissive-" )
      endif()
    else()
      list( APPEND PYTORCH_ENV_VARS_WITH_BUILD_DIR "SAM2_BUILD_CUDA=0" )
    endif()
  endif()

  string( REPLACE ";" "----" PYTORCH_ENV_VARS_STR "${PYTORCH_ENV_VARS_WITH_BUILD_DIR}" )
  string( REPLACE ";" "----" LIBRARY_PIP_BUILD_CMD_STR "${LIBRARY_PIP_BUILD_CMD}" )

  set( CONDITIONAL_BUILD_CMD
    ${CMAKE_COMMAND}
      -DLIB_NAME=${LIB}
      -DLIB_SOURCE_DIR=${LIBRARY_LOCATION}
      -DHASH_FILE=${LIB_HASH_FILE}
      -DWHEEL_DIR=${LIBRARY_PIP_BUILD_DIR}
      -DPYTHON_BUILD_CMD=${LIBRARY_PIP_BUILD_CMD_STR}
      -DENV_VARS:STRING=${PYTORCH_ENV_VARS_STR}
      -DWORKING_DIR:PATH=${LIBRARY_LOCATION} )

  # For wheel builds, have the build script also handle pip install
  if( USE_BUILD_SCRIPT_FOR_INSTALL )
    list( APPEND CONDITIONAL_BUILD_CMD
      -DPython_EXECUTABLE=${Python_EXECUTABLE}
      -DPIP_INSTALL_SCRIPT=${VIAME_CMAKE_DIR}/pip_install_with_lock.cmake
      -DNO_CACHE_DIR=${VIAME_BUILD_NO_CACHE_DIR} )
  endif()

  # mmdeploy has additional C++ build steps
  set( LIBRARY_CONFIGURE_CMD "" )
  if( "${LIB}" STREQUAL "mmdeploy" )
    set( ONNXRUNTIME_DIR ${VIAME_PYTHON_PACKAGES}/onnxruntime/onnxruntimelibs )
    set( LIBRARY_CPP_BUILD_DIR ${LIBRARY_PIP_BUILD_DIR} )
    file( MAKE_DIRECTORY ${LIBRARY_CPP_BUILD_DIR} )

    set( LIBRARY_CPP_CONFIG
      ${CMAKE_COMMAND}
      -DMMDEPLOY_TARGET_BACKENDS=ort
      -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}
      -S "${LIBRARY_LOCATION}"
      -B "${LIBRARY_CPP_BUILD_DIR}" )

    set( LIBRARY_CPP_BUILD ${CMAKE_COMMAND} --build "${LIBRARY_CPP_BUILD_DIR}" )
    set( LIBRARY_CPP_INSTALL ${CMAKE_COMMAND} --install "${LIBRARY_CPP_BUILD_DIR}" )
    if( (CMAKE_CONFIGURATION_TYPES STREQUAL "Release") OR (CMAKE_BUILD_TYPE STREQUAL "Release") )
      list( APPEND LIBRARY_CPP_BUILD --config Release )
      list( APPEND LIBRARY_CPP_INSTALL --config Release )
    endif()
    if( VIAME_BUILD_MAX_THREADS )
      list( APPEND LIBRARY_CPP_BUILD -j ${VIAME_BUILD_MAX_THREADS} )
    endif()

    # Convert C++ build/install commands to ----separated strings (like PYTHON_BUILD_CMD)
    string( REPLACE ";" "----" LIBRARY_CPP_BUILD_STR "${LIBRARY_CPP_BUILD}" )
    string( REPLACE ";" "----" LIBRARY_CPP_INSTALL_STR "${LIBRARY_CPP_INSTALL}" )

    list( APPEND CONDITIONAL_BUILD_CMD
      -DCPP_BUILD_CMD=${LIBRARY_CPP_BUILD_STR}
      -DCPP_INSTALL_CMD=${LIBRARY_CPP_INSTALL_STR} )

    set( LIBRARY_CONFIGURE_CMD ${LIBRARY_CPP_CONFIG} )
  endif()

  list( APPEND CONDITIONAL_BUILD_CMD -P ${VIAME_CMAKE_DIR}/custom_build_python_dep.cmake )

  ExternalProject_Add( ${LIB}
    DEPENDS ${PROJECT_DEPS}
    PREFIX ${VIAME_BUILD_PREFIX}
    SOURCE_DIR ${LIBRARY_LOCATION}
    BUILD_IN_SOURCE 1
    PATCH_COMMAND ${LIBRARY_PATCH_COMMAND}
    CONFIGURE_COMMAND "${LIBRARY_CONFIGURE_CMD}"
    BUILD_COMMAND ${CONDITIONAL_BUILD_CMD}
    INSTALL_COMMAND ${LIBRARY_PYTHON_INSTALL}
    LIST_SEPARATOR "----" )

  # On Windows, enable git long paths for PyTorch submodules to handle
  # composable_kernel files that exceed the 260-char MAX_PATH limit
  if( WIN32 AND "${LIB}" STREQUAL "pytorch" )
    ExternalProject_Add_Step(${LIB}
      git_longpaths
      COMMAND git config core.longpaths true
      WORKING_DIRECTORY ${LIBRARY_LOCATION}
      DEPENDEES patch
      DEPENDERS build
      COMMENT "Enabling git long paths for PyTorch submodules on Windows" )
  endif()

  if( "${LIB}" STREQUAL "mmdeploy" )
    set( MMDEPLOY_INSTALL_DIR ${VIAME_PYTHON_INSTALL}/site-packages/mmdeploy )
    ExternalProject_Add_Step(${LIB}
      postinstall
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${LIBRARY_LOCATION}/configs ${MMDEPLOY_INSTALL_DIR}/configs
      DEPENDEES install )
  endif()
endforeach()

set( VIAME_ARGS_pytorch
  -Dpytorch_DIR:PATH=${VIAME_BUILD_PREFIX}/src/pytorch-build
  )
