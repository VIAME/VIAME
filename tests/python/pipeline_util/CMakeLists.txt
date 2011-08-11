project(vistk_test_python_pipeline_util)

##############################
# Bake tests
##############################
vistk_build_python_test(bake test-bake.py)

vistk_make_python_test(bake import)

##############################
# Export tests
##############################
vistk_build_python_test(export_ test-export_.py)

vistk_make_python_test(export_ import)

##############################
# Load tests
##############################
vistk_build_python_test(load test-load.py)

vistk_make_python_test(load import)
