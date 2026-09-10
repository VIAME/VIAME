# The python bindings
#
# The package these install into. It used to come from the top-level project's
# name, which made a second copy of every extension module once kwiver was a
# subdirectory of VIAME and every VIAME module land in the `kwiver` package
# once it was not -- see design/lite-findings.md. Set here, in the one
# directory whose modules belong to it.
set( kwiver_python_package "kwiver" )

#
# A python extension module leaves the interpreter's symbols undefined and
# resolves them at import, so `-Wl,--no-undefined` cannot apply here. VIAME
# builds with it and kwiver did not, which is why this only became true when
# P5-T05 moved these out of kwiver's directory scope and into VIAME's.
string( REPLACE "-Wl,--no-undefined" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" )
string( REPLACE "-Wl,--no-undefined" "" CMAKE_SHARED_LINKER_FLAGS
  "${CMAKE_SHARED_LINKER_FLAGS}" )
string( REPLACE "-Wl,--no-undefined" "" CMAKE_MODULE_LINKER_FLAGS
  "${CMAKE_MODULE_LINKER_FLAGS}" )

# Python cmake, if testing enabled set pip to install test dependencies in venv
if (KWIVER_ENABLE_TESTS AND VENV_CREATED)
    if (PIP_COMMAND)
        python_target_add_command( "pip-install-upgrade"
                                   "${PIP_UPGRADE_COMMAND}"
                                   "Upgrading pip...")
        python_target_add_command( "pip-install-test-deps"
                                   "${PIP_INSTALL_TEST_DEPS_COMMAND}"
                                   "Pip installing Python dependencies..."
                                   "pip-install-upgrade")
    endif()
endif()

if(NOT SKBUILD)
  # TODO depend on a global python target that includes all module files,so it does not run every time
  # Create egg info for the packages mentioned in the entrypoints entry of setup.py
  add_custom_target(install_egg_info ALL
    COMMENT "Install python egg info in the build directory"
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py egg_info --egg-base ${KWIVER_BINARY_DIR}/${python_site_packages}
    )

  kwiver_install(
    DIRECTORY ${KWIVER_BINARY_DIR}/${python_site_packages}/kwiver.egg-info
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${python_site_packages}/
  )

endif()

# The `kwiver` python package: the bindings for the imported vital and
# sprokit. It keeps the package name until phase 11 renames it, and the
# binding sources keep their package tree because they cannot sit beside the
# C++ they bind while they do -- see design/lite-findings.md 1.13.
add_subdirectory( kwiver )

# Run python stub generation on installed module
if(SKBUILD)
  install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E env \"PYTHONPATH=${CMAKE_INSTALL_PREFIX}/..\" pybind11-stubgen kwiver -o \"${CMAKE_INSTALL_PREFIX}/..\")")
endif()
