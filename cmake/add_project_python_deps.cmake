# Python Dependency External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

# --------------------- ADD ANY EXTRA PYTHON DEPS HERE -------------------------

set( VIAME_PYTHON_DEPS numpy matplotlib )
set( VIAME_PYTHON_DEP_CMDS "numpy" "matplotlib" )

if( VIAME_ENABLE_TENSORFLOW )
  list( APPEND VIAME_PYTHON_DEPS humanfriendly )
  list( APPEND VIAME_PYTHON_DEP_CMDS "humanfriendly" )

  list( APPEND VIAME_PYTHON_DEPS tensorflow )
  if( VIAME_ENABLE_CUDA )
    list( APPEND VIAME_PYTHON_DEP_CMDS "tensorflow-gpu==1.14" )
  else()
    list( APPEND VIAME_PYTHON_DEP_CMDS "tensorflow==1.14" )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH-MMDET )
  list( APPEND VIAME_PYTHON_DEPS yapf )
  list( APPEND VIAME_PYTHON_DEP_CMDS "yapf" )
endif()

if( VIAME_ENABLE_CAMTRAWL OR VIAME_ENABLE_OPENCV )
  list( APPEND VIAME_PYTHON_DEPS tqdm scipy )
  list( APPEND VIAME_PYTHON_DEP_CMDS "tqdm" "scipy" )
endif()

if( ( WIN32 OR NOT VIAME_ENABLE_OPENCV ) AND
      ( VIAME_ENABLE_CAMTRAWL OR VIAME_ENABLE_OPENCV OR
        VIAME_ENABLE_PYTORCH-MMDET OR VIAME_ENABLE_PYTORCH-NETHARN ) )
  list( APPEND VIAME_PYTHON_DEPS cv2 )
  list( APPEND VIAME_PYTHON_DEP_CMDS "opencv-python" )
endif()

if( VIAME_ENABLE_ITK_EXTRAS )
  set( WX_VERSION "4.0.7" )

  list( APPEND VIAME_PYTHON_DEPS msgpack )
  list( APPEND VIAME_PYTHON_DEP_CMDS "msgpack" )

  list( APPEND VIAME_PYTHON_DEPS wxPython )

  if( UNIX )
    if( EXISTS "/etc/os-release" )
      ParseLinuxOSField( "ID" OS_ID )
    endif()

    execute_process( COMMAND lsb_release -cs
      OUTPUT_VARIABLE RELEASE_CODENAME
      OUTPUT_STRIP_TRAILING_WHITESPACE )

    if( "${OS_ID}" MATCHES "centos" )
      set( WXP_ARCHIVE https://extras.wxpython.org/wxPython4/extras/linux/gtk3/centos-7 )
      list( APPEND VIAME_PYTHON_DEP_CMDS "-U -f ${WXP_ARCHIVE} wxPython==${WX_VERSION}" )
    elseif( "${RELEASE_CODENAME}" MATCHES "xenial" )
      set( WXP_ARCHIVE https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04 )
      list( APPEND VIAME_PYTHON_DEP_CMDS "-U -f ${WXP_ARCHIVE} wxPython==${WX_VERSION}" )
    else()
      list( APPEND VIAME_PYTHON_DEP_CMDS "wxPython==${WX_VERSION}" )
    endif()
  else()
    list( APPEND VIAME_PYTHON_DEP_CMDS "wxPython==${WX_VERSION}" )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH )
  list( APPEND VIAME_PYTHON_DEPS scikit-image )
  list( APPEND VIAME_PYTHON_DEP_CMDS "scikit-image==0.16.2" )
endif()

if( VIAME_ENABLE_CAMTRAWL OR VIAME_ENABLE_PYTORCH-NETHARN )
  list( APPEND VIAME_PYTHON_DEPS imgaug ubelt pygments bezier )
  list( APPEND VIAME_PYTHON_DEP_CMDS "imgaug" "ubelt" "pygments" "bezier==2020.1.14" )
endif()

if( VIAME_ENABLE_PYTORCH AND VIAME_ENABLE_PYTORCH-INTERNAL )
  list( APPEND VIAME_PYTHON_DEPS "pyyaml" )
  list( APPEND VIAME_PYTHON_DEP_CMDS "pyyaml" )
endif()

if( VIAME_ENABLE_PYTORCH AND VIAME_ENABLE_PYTORCH-MMDET AND NOT WIN32 )
  list( APPEND VIAME_PYTHON_DEPS "pycocotools" )
  list( APPEND VIAME_PYTHON_DEP_CMDS "pycocotools" )
endif()

if( VIAME_ENABLE_PYTORCH AND NOT VIAME_ENABLE_PYTORCH-INTERNAL )
  list( APPEND VIAME_PYTHON_DEPS pytorch )

  set( ARGS_TORCH )
  set( PYTORCH_ARCHIVE https://download.pytorch.org/whl/torch_stable.html )
  set( PYTORCH_VERSION 1.4.0 )

  if( WIN32 AND VIAME_ENABLE_CUDA )
    if( CUDA_VERSION VERSION_EQUAL "10.1" )
      if( PYTHON_VERSION VERSION_EQUAL 3.6 )
        set( ARGS_TORCH "https://download.pytorch.org/whl/cu101/torch-1.4.0-cp36-cp36m-win_amd64.whl" )
      else()
        set( ARGS_TORCH "===${PYTORCH_VERSION} -f ${PYTORCH_ARCHIVE}" )
      endif()
    elseif( CUDA_VERSION VERSION_EQUAL "9.2" )
      set( ARGS_TORCH "==${PYTORCH_VERSION}+cu92 -f ${PYTORCH_ARCHIVE}" )
    else()
      message( FATAL_ERROR "With your current build settings you must either:\n"
        " (a) Turn on VIAME_ENABLE_PYTORCH-INTERNAL\n"
        " (b) Use CUDA 9.2, or 10.1\n"
        " (c) Disable VIAME_ENABLE_PYTORCH\n" )
    endif()
  elseif( VIAME_ENABLE_CUDA )
    if( CUDA_VERSION VERSION_EQUAL "10.1" )
      set( ARGS_TORCH "===${PYTORCH_VERSION} -f ${PYTORCH_ARCHIVE}" )
    elseif( CUDA_VERSION VERSION_EQUAL "10.0" )
      set( ARGS_TORCH "==${PYTORCH_VERSION}+cu100 -f ${PYTORCH_ARCHIVE}" )
    elseif( CUDA_VERSION VERSION_EQUAL "9.2" )
      set( ARGS_TORCH "==${PYTORCH_VERSION}+cu92 -f ${PYTORCH_ARCHIVE}" )
    else()
      message( FATAL_ERROR "With your current build settings you must either:\n"
        " (a) Turn on VIAME_ENABLE_PYTORCH-INTERNAL\n"
        " (b) Use CUDA 9.2, 10.0, or 10.1\n"
        " (c) Disable VIAME_ENABLE_PYTORCH\n" )
    endif()
  else()
    set( ARGS_TORCH "==${PYTORCH_VERSION}+cpu -f ${PYTORCH_ARCHIVE}" )
  endif()

  string( FIND "${ARGS_TORCH}" "https://" TMP_VAR )
  if( "${TMP_VAR}" EQUAL 0 )
    list( APPEND VIAME_PYTHON_DEP_CMDS "${ARGS_TORCH}" )
  else()
    list( APPEND VIAME_PYTHON_DEP_CMDS "torch${ARGS_TORCH}" )
  endif()
endif()

# ------------------------------------------------------------------------------

set( PYTHON_BASEPATH
  ${VIAME_BUILD_INSTALL_PREFIX}/lib/python${PYTHON_VERSION} )

if( WIN32 )
  set( CUSTOM_PYTHONPATH
    ${PYTHON_BASEPATH};${PYTHON_BASEPATH}/site-packages;${PYTHON_BASEPATH}/dist-packages )
  set( CUSTOM_PATH
    ${VIAME_BUILD_INSTALL_PREFIX}/bin;$ENV{PATH} )

  if( VIAME_ENABLE_PYTHON-INTERNAL )
    set( ENV{PYTHONPATH} "${CUSTOM_PYTHONPATH};ENV{PYTHONPATH}" )
  endif()

  string( REPLACE ";" "----" CUSTOM_PYTHONPATH "${CUSTOM_PYTHONPATH}" )
  string( REPLACE ";" "----" CUSTOM_PATH "${CUSTOM_PATH}" )
else()
  set( CUSTOM_PYTHONPATH
    ${PYTHON_BASEPATH}:${PYTHON_BASEPATH}/site-packages:${PYTHON_BASEPATH}/dist-packages )
  set( CUSTOM_PATH
    ${VIAME_BUILD_INSTALL_PREFIX}/bin:$ENV{PATH} )
endif()

set( VIAME_PYTHON_DEPS_DEPS fletch )

if( VIAME_ENABLE_SMQTK )
  set( VIAME_PYTHON_DEPS_DEPS smqtk ${VIAME_PYTHON_DEPS_DEPS} )
endif()

list( LENGTH VIAME_PYTHON_DEPS DEP_COUNT )
math( EXPR DEP_COUNT "${DEP_COUNT} - 1" )

foreach( ID RANGE ${DEP_COUNT} )

  list( GET VIAME_PYTHON_DEPS ${ID} DEP )
  list( GET VIAME_PYTHON_DEP_CMDS ${ID} CMD )

  set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} ${DEP} )
  set( PYTHON_LIB_DEPS ${VIAME_PYTHON_DEPS_DEPS} )

  if( "${DEP}" STREQUAL "imgaug" AND VIAME_ENABLE_PYTORCH )
    set( PYTHON_LIB_DEPS ${VIAME_PYTHON_DEPS_DEPS} scikit-image )
  else()
    set( PYTHON_LIB_DEPS ${VIAME_PYTHON_DEPS_DEPS} )
  endif()

  set( PYTHON_DEP_PIP_CMD pip install --user ${CMD} )
  string( REPLACE " " ";" PYTHON_DEP_PIP_CMD "${PYTHON_DEP_PIP_CMD}" )

  set( PYTHON_DEP_INSTALL
    ${CMAKE_COMMAND} -E env "PYTHONPATH=${CUSTOM_PYTHONPATH}"
                            "PATH=${CUSTOM_PATH}"
                            "PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}"
      ${PYTHON_EXECUTABLE} -m ${PYTHON_DEP_PIP_CMD}
    )

  ExternalProject_Add( ${DEP}
    DEPENDS ${PYTHON_LIB_DEPS}
    PREFIX ${VIAME_BUILD_PREFIX}
    SOURCE_DIR ${VIAME_CMAKE_DIR}
    USES_TERMINAL_BUILD 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${PYTHON_DEP_INSTALL}
    INSTALL_COMMAND ""
    INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
    LIST_SEPARATOR "----"
    )
endforeach()
