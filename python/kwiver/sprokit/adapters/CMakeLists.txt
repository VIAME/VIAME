project(sprokit_python_sprokit_adapters)

viame_add_python_library(adapter_data_set
  sprokit/adapters
  SOURCES adapter_data_set.cxx)

viame_target_link_libraries(python-sprokit.adapters-adapter_data_set
  PRIVATE             viame_adapter
                      viame_pipeline_framework
                      viame_python_util)


viame_add_python_library(embedded_pipeline
  sprokit/adapters
  SOURCES embedded_pipeline.cxx)

viame_target_link_libraries(python-sprokit.adapters-embedded_pipeline
  PRIVATE             viame_adapter
                      viame_pipeline_util
                      viame_python_util)

viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  sprokit/adapters
  __init__)
