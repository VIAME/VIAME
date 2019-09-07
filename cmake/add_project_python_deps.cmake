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

if( VIAME_ENABLE_CAMTRAWL )
  list( APPEND VIAME_PYTHON_DEPS ubelt )
  list( APPEND VIAME_PYTHON_DEP_CMDS "ubelt" )
endif()

if( VIAME_ENABLE_TENSORFLOW )
  list( APPEND VIAME_PYTHON_DEPS humanfriendly )
  list( APPEND VIAME_PYTHON_DEP_CMDS "humanfriendly" )

  list( APPEND VIAME_PYTHON_DEPS tensorflow )
  if( VIAME_ENABLE_CUDA )
    list( APPEND VIAME_PYTHON_DEP_CMDS "tensorflow-gpu" )
  else()
    list( APPEND VIAME_PYTHON_DEP_CMDS "tensorflow" )
  endif()
endif()

if( VIAME_ENABLE_ITK )
  list( APPEND VIAME_PYTHON_DEPS msgpack )
  list( APPEND VIAME_PYTHON_DEP_CMDS "msgpack" )

  list( APPEND VIAME_PYTHON_DEPS wxPython )

  if( UNIX )
    string( REGEX MATCH "\\.el[1-9]" OS_RHEL_SUFFIX ${CMAKE_SYSTEM} )

    execute_process( COMMAND lsb_release -cs
      OUTPUT_VARIABLE RELEASE_CODENAME
      OUTPUT_STRIP_TRAILING_WHITESPACE )

    if( "${OS_RHEL_SUFFIX}" MATCHES ".el7*" )
      set( WXP_ARCHIVE https://extras.wxpython.org/wxPython4/extras/linux/gtk3/centos-7 )
      list( APPEND VIAME_PYTHON_DEP_CMDS "-U -f ${WXP_ARCHIVE} wxPython" )
    elseif( "${RELEASE_CODENAME}" MATCHES "xenial" )
      set( WXP_ARCHIVE https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04 )
      list( APPEND VIAME_PYTHON_DEP_CMDS "-U -f ${WXP_ARCHIVE} wxPython" )
    else()
      list( APPEND VIAME_PYTHON_DEP_CMDS "wxPython" )
    endif()
  else()
    list( APPEND VIAME_PYTHON_DEP_CMDS "wxPython" )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH AND NOT VIAME_ENABLE_PYTORCH-INTERNAL )
  list( APPEND VIAME_PYTHON_DEPS torch )
  list( APPEND VIAME_PYTHON_DEPS torchvision )

  set( ARGS_TORCH )
  set( ARGS_TORCHVISION )

  set( PYTORCH_ARCHIVE https://download.pytorch.org/whl/torch_stable.html )

  if( WIN32 )
    if( CUDA_VERSION VERSION_GREATER_EQUAL "10.0" )
      set( ARGS_TORCH ==1.2.0 -f ${PYTORCH_ARCHIVE} )
      set( ARGS_TORCHVISION ==0.4.0 -f ${PYTORCH_ARCHIVE} )
    elseif( CUDA_VERSION VERSION_EQUAL "9.2" )
      set( ARGS_TORCH ==1.2.0+cu92 -f ${PYTORCH_ARCHIVE} )
      set( ARGS_TORCHVISION ==0.4.0+cu92 -f ${PYTORCH_ARCHIVE} )
    else()
      message( FATAL_ERROR "With your current build settings you must either:\n"
        " (a) Turn on VIAME_ENABLE_PYTORCH-INTERNAL or\n"
        " (b) Use CUDA 9.2 or 10.0+\n"
        " (c) Disable VIAME_ENABLE_PYTORCH\n" )
    endif()
  else()
    if( CUDA_VERSION VERSION_EQUAL "9.2" )
      set( ARGS_TORCH ==1.2.0+cu92 -f ${PYTORCH_ARCHIVE} )
      set( ARGS_TORCHVISION ==0.4.0+cu92 -f ${PYTORCH_ARCHIVE} )
    elseif( CUDA_VERSION VERSION_LESS "10.0" )
      message( FATAL_ERROR "With your current build settings you must either:\n"
        " (a) Turn on VIAME_ENABLE_PYTORCH-INTERNAL or\n"
        " (b) Use CUDA 9.2 or 10.0+\n"
        " (c) Disable VIAME_ENABLE_PYTORCH\n" )
    endif()
  endif()

  list( APPEND VIAME_PYTHON_DEP_CMDS "torch${ARGS_TORCH}" )
  list( APPEND VIAME_PYTHON_DEP_CMDS "torchvision${ARGS_TORCHVISION}" )
endif()

# ------------------------------------------------------------------------------

set( PYTHON_BASEPATH
  ${VIAME_BUILD_INSTALL_PREFIX}/lib/python${PYTHON_VERSION} )

if( WIN32 )
  set( CUSTOM_PYTHONPATH
    ${PYTHON_BASEPATH}/site-packages;${PYTHON_BASEPATH}/dist-packages )
  set( CUSTOM_PATH
    ${VIAME_BUILD_INSTALL_PREFIX}/bin;$ENV{PATH} )

  string( REPLACE ";" "----" CUSTOM_PYTHONPATH "${CUSTOM_PYTHONPATH}" )
  string( REPLACE ";" "----" CUSTOM_PATH "${CUSTOM_PATH}" )
else()
  set( CUSTOM_PYTHONPATH
    ${PYTHON_BASEPATH}/site-packages:${PYTHON_BASEPATH}/dist-packages )
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

  set( PYTHON_DEP_PIP_CMD pip install --user ${CMD} )
  string( REPLACE " " ";" PYTHON_DEP_PIP_CMD "${PYTHON_DEP_PIP_CMD}" )

  set( PYTHON_DEP_INSTALL
    ${CMAKE_COMMAND} -E env "PYTHONPATH=${CUSTOM_PYTHONPATH}"
                            "PATH=${CUSTOM_PATH}"
                            "PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}"
      ${PYTHON_EXECUTABLE} -m ${PYTHON_DEP_PIP_CMD}
    )

  ExternalProject_Add( ${DEP}
    DEPENDS ${VIAME_PYTHON_DEPS_DEPS}
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
