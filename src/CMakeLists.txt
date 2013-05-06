include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_subdirectory(sprokit)

option(SPROKIT_ENABLE_TOOLS "Build tools" ON)
if (SPROKIT_ENABLE_TOOLS)
  add_subdirectory(tools)
endif ()

function (add_sprokit_plugin name define)
  set(library_subdir /sprokit)

  set(no_export ON)

  sprokit_add_library(${name} SHARED
    ${ARGN})

  set_target_properties(${name}
    PROPERTIES
      DEFINE_SYMBOL    ${define}
      PREFIX           ""
      SUFFIX           ${CMAKE_SHARED_MODULE_SUFFIX})
endfunction ()

add_subdirectory(bindings)
add_subdirectory(processes)
add_subdirectory(schedulers)
