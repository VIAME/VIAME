set( core_python_headers
     render_mesh_depth_map.h
   )

 set( core_python_sources
      render_mesh_depth_map.cxx
      core_module.cxx
    )

kwiver_add_python_library( core
   arrows/core
   ${core_python_headers}
   ${core_python_sources}
)

target_link_libraries( python-arrows.core-core
  PUBLIC ${PYTHON_LIBRARIES}
         vital
         kwiver_algo_core
)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                                arrows/core
                                __init__)
