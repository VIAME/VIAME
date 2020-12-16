project(sprokit_python_sprokit_adapters)

kwiver_add_python_library(adapter_data_set
  sprokit/adapters
  adapter_data_set.cxx)

target_link_libraries(python-sprokit.adapters-adapter_data_set
  PRIVATE             kwiver_adapter
                      sprokit_pipeline
                      vital_python_util)


kwiver_add_python_library(embedded_pipeline
  sprokit/adapters
  embedded_pipeline.cxx)

target_link_libraries(python-sprokit.adapters-embedded_pipeline
  PRIVATE             kwiver_adapter
                      sprokit_pipeline_util
                      vital_python_util)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  sprokit/adapters
  __init__)
