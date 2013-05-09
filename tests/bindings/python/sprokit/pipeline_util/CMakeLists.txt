project(sprokit_test_python_sprokit_pipeline_util)

set(sprokit_test_pipelines_directory
  ${sprokit_test_data_directory}/pipelines)

function (sprokit_add_python_load_test name instance)
  sprokit_add_python_test(${name} ${instance}
    "${sprokit_test_pipelines_directory}")
endfunction ()

##############################
# Bake tests
##############################
sprokit_build_python_test(bake test-bake.py)

sprokit_add_python_load_test(bake import)
sprokit_add_python_load_test(bake simple_pipeline)
sprokit_add_python_load_test(bake cluster_multiplier)

##############################
# Export tests
##############################
sprokit_build_python_test(export_ test-export_.py)

sprokit_add_python_load_test(export_ import)
sprokit_add_python_load_test(export_ simple_pipeline)

##############################
# Load tests
##############################
sprokit_build_python_test(load test-load.py)

sprokit_add_python_load_test(load import)
sprokit_add_python_load_test(load create)
sprokit_add_python_load_test(load api_calls)
sprokit_add_python_load_test(load simple_pipeline)
sprokit_add_python_load_test(load cluster_multiplier)
