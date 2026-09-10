set( THIS_MODULE vital/tests/cpp_helpers)
set(__helper_modnames
  camera_helpers
  camera_intrinsics_helpers
  transform_2d_helpers
  det_obj_set_helpers
  camera_perspective_helpers
  camera_perspective_map_helpers
  camera_rpc_helpers
  metadata_map_helpers
  type_check
  )

# Add files to the vital.tests.cpp_helpers python module
foreach (modname IN LISTS __helper_modnames)
  kwiver_add_python_library(
    ${modname}
    "${THIS_MODULE}"
    SOURCES ${modname}.cxx
    PRIVATE vital
    )

endforeach()

if(NOT SKBUILD)
    kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                              vital/tests/cpp_helpers
                              __init__)
endif()
