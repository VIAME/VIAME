# Python cmake, if testing enabled set pip to install test dependencies in venv
if (KWIVER_ENABLE_TESTS)
    if (PIP_COMMAND)
        list(SUBLIST PIP_COMMAND 0 1 py_pip_command)
        list(SUBLIST PIP_COMMAND 1 -1 pip_args)
        python_target_add_command( "pip-install-upgrade"
                                   ${py_pip_command}
                                   "${PIP_UPGRADE_COMMAND}"
                                   "Upgrading pip...")
        python_target_add_command( "pip"
                                   ${py_pip_command}
                                   "${pip_args}"
                                   "Pip installing Python dependencies..."
                                   "pip-install-upgrade")
    endif()
endif()
add_subdirectory(kwiver)
