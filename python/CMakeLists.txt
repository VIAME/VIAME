# Python cmake, if testing enabled set pip to install test dependencies in venv
if (KWIVER_ENABLE_TESTS)
    if (PIP_COMMAND)
        python_target_add_command( "pip-install-upgrade"
                                   "${PIP_UPGRADE_COMMAND}"
                                   "Upgrading pip...")
        python_target_add_command( "pip-install-test-deps"
                                   "${PIP_INSTALL_COMMAND}"
                                   "Pip installing Python dependencies..."
                                   "pip-install-upgrade")
    endif()
endif()
add_subdirectory(kwiver)
