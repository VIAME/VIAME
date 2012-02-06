project(vistk_test_python_modules)

##############################
# Config tests
##############################
vistk_build_python_test(pymodules test-pymodules.py)

vistk_make_python_test(pymodules import)
vistk_make_python_test(pymodules load)
vistk_make_python_test(pymodules masking)
vistk_make_python_test(pymodules extra_modules)

set_tests_properties(test-python-pymodules-masking
  PROPERTIES
    ENVIRONMENT "VISTK_NO_PYTHON_MODULES=")

set_tests_properties(test-python-pymodules-extra_modules
  PROPERTIES
    ENVIRONMENT "VISTK_PYTHON_MODULES=vistk.test.python.modules")
