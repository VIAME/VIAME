project(sprokit_python_sprokit_pipeline_util)

if( NOT WIN32 )

viame_add_python_library(bake
  sprokit/pipeline_util
  SOURCES bake.cxx)

viame_target_link_libraries(python-sprokit.pipeline_util-bake
  LINK_PRIVATE
    viame_pipeline_framework
    viame_pipeline_util
    viame_pipeline_python_util
    viame_python_util
    ${Boost_IOSTREAMS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    viame_plugin
  )

viame_add_python_library(load
  sprokit/pipeline_util
  SOURCES load.cxx)
viame_target_link_libraries(python-sprokit.pipeline_util-load
  LINK_PRIVATE
    viame_pipeline_framework
    viame_pipeline_util
    viame_pipeline_python_util
    viame_python_util
    viame_util
    ${Boost_IOSTREAMS_LIBRARY})

viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                         sprokit/pipeline_util
                         __init__)
endif()
