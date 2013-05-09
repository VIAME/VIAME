project(sprokit_test_python_modules)

##############################
# Config tests
##############################
sprokit_build_python_test(pymodules test-pymodules.py)

sprokit_add_python_test(pymodules import)
sprokit_add_python_test(pymodules load)
set(sprokit_test_environment "SPROKIT_NO_PYTHON_MODULES=")
sprokit_add_python_test(pymodules masking)
set(sprokit_test_environment "SPROKIT_PYTHON_MODULES=sprokit.test.python.modules")
sprokit_add_python_test(pymodules extra_modules)
set(sprokit_test_environment "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}")
sprokit_add_python_test(pymodules pythonpath)
