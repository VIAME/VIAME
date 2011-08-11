project(vistk_test_python_pipeline_util)

##############################
# Bake tests
##############################
vistk_build_python_test(bake test-python-bake.py.in)

vistk_make_python_test(bake import)

##############################
# Export tests
##############################
vistk_build_python_test(export_ test-python-export_.py.in)

vistk_make_python_test(export_ import)

##############################
# Load tests
##############################
vistk_build_python_test(load test-python-load.py.in)

vistk_make_python_test(load import)
