include_directories(${pybind11_INCLUDE_DIR})

set(__helper_modnames
  camera_helpers
  camera_intrinsics_helpers
  transform_2d_helpers
  det_obj_set_helpers
  camera_perspective_helpers
  camera_perspective_map_helpers
  camera_rpc_helpers
  metadata_map_helpers
  )

# Add files to the vital.tests.cpp_helpers python module
foreach (modname IN LISTS __helper_modnames)
  kwiver_add_python_library(${modname}
    vital/tests/cpp_helpers
    ${modname}.cxx)

  target_link_libraries(python-vital.tests.cpp_helpers-${modname}
    PUBLIC ${PYTHON_LIBRARIES}
           vital)

  if(NOT SKBUILD)
    kwiver_create_python_init(vital/tests/cpp_helpers
      ${modname}
    )
  endif()
endforeach()
