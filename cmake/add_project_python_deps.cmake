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

if( VIAME_ENABLE_OPENCV )
  set( VIAME_PYTHON_DEPS opencv-python ${VIAME_PYTHON_DEPS} )
endif()

if( VIAME_ENABLE_CAMTRAWL )
  set( VIAME_PYTHON_DEPS ubelt ${VIAME_PYTHON_DEPS} )
endif()

if( VIAME_ENABLE_TENSORFLOW )
  if( VIAME_ENABLE_CUDA )
    set( VIAME_PYTHON_DEPS tensorflow-gpu humanfriendly ${VIAME_PYTHON_DEPS} )
  else()
    set( VIAME_PYTHON_DEPS tensorflow humanfriendly ${VIAME_PYTHON_DEPS} )
  endif()

  set( VIAME_PIP_ARGS_TENSORFLOW )
endif()

if( VIAME_ENABLE_PYTORCH AND NOT VIAME_ENABLE_PYTORCH-INTERNAL )
  set( VIAME_PYTHON_DEPS torch torchvision ${VIAME_PYTHON_DEPS} )

  set( VIAME_PIP_ARGS_TORCH )
  set( VIAME_PIP_ARGS_TORCHVISION )

  set( PYTORCH_ARCHIVE https://download.pytorch.org/whl/torch_stable.html )

  if( WIN32 )
    if( CUDA_VERSION VERSION_GREATER_EQUAL "10.0" )
      set( VIAME_PIP_ARGS_TORCH ==1.2.0 -f ${PYTORCH_ARCHIVE} )
      set( VIAME_PIP_ARGS_TORCHVISION ==0.4.0 -f ${PYTORCH_ARCHIVE} )
    elseif( CUDA_VERSION VERSION_EQUAL "9.2" )
      set( VIAME_PIP_ARGS_TORCH ==1.2.0+cu92 -f ${PYTORCH_ARCHIVE} )
      set( VIAME_PIP_ARGS_TORCHVISION ==0.4.0+cu92 -f ${PYTORCH_ARCHIVE} )
    else()
      message( FATAL_ERROR "With your current build settings you must either:\n"
        " (a) Turn on VIAME_ENABLE_PYTORCH-INTERNAL or\n"
        " (b) Use CUDA 9.2 or 10.0+\n"
        " (c) Disable VIAME_ENABLE_PYTORCH\n" )
    endif()
  else()
    if( CUDA_VERSION VERSION_EQUAL "9.2" )
      set( VIAME_PIP_ARGS_TORCH ==1.2.0+cu92 -f ${PYTORCH_ARCHIVE} )
      set( VIAME_PIP_ARGS_TORCHVISION ==0.4.0+cu92 -f ${PYTORCH_ARCHIVE} )
    elseif( CUDA_VERSION VERSION_LESS "10.0" )
      message( FATAL_ERROR "With your current build settings you must either:\n"
        " (a) Turn on VIAME_ENABLE_PYTORCH-INTERNAL or\n"
        " (b) Use CUDA 9.2 or 10.0+\n"
        " (c) Disable VIAME_ENABLE_PYTORCH\n" )
    endif()
  endif()
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

foreach( DEP ${VIAME_PYTHON_DEPS} )

  set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} ${DEP} )

  if( "${DEP}" STREQUAL "torch" )
    set( DEPARGS ${VIAME_PIP_ARGS_TORCH} )
  elseif( "${DEP}" STREQUAL "torchvision" )
    set( DEPARGS ${VIAME_PIP_ARGS_TORCHVISION} )
  elseif( "${DEP}" STREQUAL "tensorflow" OR "${DEP}" STREQUAL "tensorflow-gpu" )
    set( DEPARGS ${VIAME_PIP_ARGS_TENSORFLOW} )
  else()
    set( DEPARGS )
  endif()

  set( PYTHON_DEP_PIP_CMD
      pip install --user ${DEP}${DEPARGS} )

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
