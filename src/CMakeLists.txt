include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_subdirectory(vistk)

if (VISTK_ENABLE_TOOLS)
  add_subdirectory(tools)
endif (VISTK_ENABLE_TOOLS)

function (add_vistk_plugin name define)
  set(library_subdir /vistk)

  vistk_add_library(${name} SHARED
    ${ARGN})

  set_target_properties(${name}
    PROPERTIES
      DEFINE_SYMBOL    ${define}
      PREFIX           "")
endfunction (add_vistk_plugin name define)

add_subdirectory(bindings)
add_subdirectory(processes)
add_subdirectory(schedulers)
