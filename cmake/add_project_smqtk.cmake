# vibrant External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} smqtk )

if( WIN32 )
  message( FATAL_ERROR "SMQTK not yet supported on WIN32" )
else()

  # Logic for getting site packages is copied from kwiver
  # (but it uses a new pycmd function that handles indentation -- ooohh. aahh.)
  pycmd(python_site_packages "
    from distutils import sysconfig
    print(sysconfig.get_python_lib(prefix=''))
  ")
  message(STATUS "python_site_packages = ${python_site_packages}")
  get_filename_component(python_sitename ${python_site_packages} NAME)

  pycmd(python_pip_version "import pip; print(pip.__version__)")

  set( PYTHON_BASEPATH
    ${VIAME_BUILD_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}${PYTHON_ABIFLAGS} )
  set( CUSTOM_SITEPACKAGES ${PYTHON_BASEPATH}/${python_sitename})
  set( CUSTOM_PYTHONPATH ${CUSTOM_SITEPACKAGES}:$ENV{PYTHONPATH} )
  set( CUSTOM_PATH ${VIAME_BUILD_INSTALL_PREFIX}/bin:$ENV{PATH} )

  # NOTES: In more recent pip versions, this step is much easier
  # https://pip.pypa.io/en/stable/news/
  # Useful commands and the versions they were added in:
  # version 0.1.4: --install-option
  # version 1.1.0: --target
  # version 1.3.0: --root
  # version 7.0.0: --prefix
  # version ~8.0.0: fixes issues with no -e and "extras_require"

  if (python_pip_version VERSION_LESS 9.0.0)
    if( VIAME_SYMLINK_PYTHON )
      set( SMQTK_PIP_CMD
        pip install --user -e .[postgres] )
    else()
      # This is only required for no symlink install without a -e with older
      # versions of pip, for never versions the above command works with no -e
      set( SMQTK_PIP_CMD
        pip install --user file://${VIAME_PACKAGES_DIR}/smqtk\#egg=smqtk[postgres] )
    endif()
    set( SMQTK_PYTHON_INSTALL
      ${CMAKE_COMMAND} -E env PYTHONPATH=${CUSTOM_PYTHONPATH}
                          env PATH=${CUSTOM_PATH}
                          env PYTHONUSERBASE=${VIAME_BUILD_INSTALL_PREFIX}
        ${PYTHON_EXECUTABLE} -m ${SMQTK_PIP_CMD} )
  else()
    # If we are using virtualenvs then a more up to date pip is necessary
    if( VIAME_SYMLINK_PYTHON )
      set( SMQTK_PIP_CMD pip install "--prefix=${VIAME_BUILD_INSTALL_PREFIX}" -e .[postgres])
    else()
      set( SMQTK_PIP_CMD pip install "--prefix=${VIAME_BUILD_INSTALL_PREFIX}" .[postgres])
    endif()
    set( SMQTK_PYTHON_INSTALL
      ${CMAKE_COMMAND} -E env PYTHONPATH=${CUSTOM_PYTHONPATH}
                          env PATH=${CUSTOM_PATH}
        ${PYTHON_EXECUTABLE} -m ${SMQTK_PIP_CMD} )
  endif()
endif()

ExternalProject_Add( smqtk
  DEPENDS fletch
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/smqtk
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    -DSMQTK_BUILD_EXAMPLES:BOOL=OFF
    -DSMQTK_BUILD_LIBSVM:BOOL=ON
    -DSMQTK_INSTALL_SETUP_SCRIPT:BOOL=OFF
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}

  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  )

ExternalProject_Add_Step(smqtk installpy
  WORKING_DIRECTORY ${VIAME_PACKAGES_DIR}/smqtk
  COMMAND ${SMQTK_PYTHON_INSTALL}
  COMMENT "Installing SMQTK python files."
  DEPENDEES build
  )

ExternalProject_Add_Step(smqtk forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${VIAME_BUILD_PREFIX}/src/smqtk-stamp/smqtk-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )

set(VIAME_ARGS_smqtk
  -Dsmqtk_DIR:PATH=${VIAME_BUILD_PREFIX}/src/smqtk-build
  )
