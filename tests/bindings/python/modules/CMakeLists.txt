project(sprokit_test_python_modules)

##############################
# Config tests
##############################
sprokit_build_python_test(pymodules test-pymodules.py)

sprokit_make_python_test(pymodules import)
sprokit_make_python_test(pymodules load)
set(test_environment "SPROKIT_NO_PYTHON_MODULES=")
sprokit_make_python_test(pymodules masking)
set(test_environment "SPROKIT_PYTHON_MODULES=sprokit.test.python.modules")
sprokit_make_python_test(pymodules extra_modules)
set(test_environment "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}")
sprokit_make_python_test(pymodules pythonpath)
