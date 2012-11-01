project(vistk_test_python_vistk_pipeline_util)

set(vistk_test_pipelines_directory
  ${vistk_test_data_directory}/pipelines)

function (vistk_make_python_load_test name instance)
  vistk_make_python_test(${name} ${instance}
    "${vistk_test_pipelines_directory}")
endfunction ()

##############################
# Bake tests
##############################
vistk_build_python_test(bake test-bake.py)

vistk_make_python_load_test(bake import)
vistk_make_python_load_test(bake simple_pipeline)

##############################
# Export tests
##############################
vistk_build_python_test(export_ test-export_.py)

vistk_make_python_load_test(export_ import)
vistk_make_python_load_test(export_ simple_pipeline)

##############################
# Load tests
##############################
vistk_build_python_test(load test-load.py)

vistk_make_python_load_test(load import)
vistk_make_python_load_test(load create)
vistk_make_python_load_test(load api_calls)
vistk_make_python_load_test(load simple_pipeline)
vistk_make_python_load_test(load cluster_multiplier)
