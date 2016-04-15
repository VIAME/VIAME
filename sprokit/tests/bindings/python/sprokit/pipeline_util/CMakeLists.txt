project(sprokit_test_python_sprokit_pipeline_util)

set(sprokit_test_pipelines_directory
  "${sprokit_test_data_directory}/pipelines")

##############################
# Bake tests
##############################
sprokit_discover_python_tests(bake test-bake.py
  "${sprokit_test_pipelines_directory}")

##############################
# Export tests
##############################
sprokit_discover_python_tests(export_ test-export_.py
  "${sprokit_test_pipelines_directory}")

##############################
# Load tests
##############################
sprokit_discover_python_tests(load test-load.py
  "${sprokit_test_pipelines_directory}")
