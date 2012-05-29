project(vistk_test_python_modules)

##############################
# Config tests
##############################
vistk_build_python_test(pymodules test-pymodules.py)

vistk_make_python_test(pymodules import)
vistk_make_python_test(pymodules load)
set(test_environment "VISTK_NO_PYTHON_MODULES=")
vistk_make_python_test(pymodules masking)
set(test_environment "VISTK_PYTHON_MODULES=vistk.test.python.modules")
vistk_make_python_test(pymodules extra_modules)
set(test_environment "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}")
vistk_make_python_test(pymodules pythonpath)
