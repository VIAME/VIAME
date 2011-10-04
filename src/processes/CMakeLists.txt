project(vistk_processes)

add_subdirectory(helpers)
add_subdirectory(examples)
add_subdirectory(flow)

if (VXL_FOUND)
  include_directories(SYSTEM ${VXL_CORE_INCLUDE_DIR})
  include_directories(SYSTEM ${VXL_VCL_INCLUDE_DIR})

  link_directories(${VXL_LIBRARY_DIR})

  add_subdirectory(image)
endif (VXL_FOUND)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    ${CMAKE_CURRENT_SOURCE_DIR}
    processes
    pipeline
    pipeline_types)
endif (VISTK_ENABLE_DOCUMENTATION)
