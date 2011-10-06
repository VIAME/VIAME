project(vistk_processes_helpers)

if (VXL_FOUND)
  include_directories(SYSTEM ${VXL_CORE_INCLUDE_DIR})
  include_directories(SYSTEM ${VXL_VCL_INCLUDE_DIR})

  link_directories(${VXL_LIBRARY_DIR})

  add_subdirectory(image)
endif (VXL_FOUND)
