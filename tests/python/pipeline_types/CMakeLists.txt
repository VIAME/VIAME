project(vistk_test_python_pipeline_types)

##############################
# Basic tests
##############################
vistk_build_python_test(basic test-basic.py)

vistk_make_python_test(basic import)
vistk_make_python_test(basic api_calls)

##############################
# Image tests
##############################
vistk_build_python_test(image test-image.py)

vistk_make_python_test(image import)
vistk_make_python_test(image api_calls)

##############################
# Utility tests
##############################
vistk_build_python_test(utility test-utility.py)

vistk_make_python_test(utility import)
vistk_make_python_test(utility api_calls)
