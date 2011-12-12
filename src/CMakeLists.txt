include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(vistk)

function (add_vistk_plugin name define)
  add_library(${name} SHARED
    ${ARGN})

  if (WIN32)
    set(module_output RUNTIME_OUTPUT_DIRECTORY)
  else (WIN32)
    set(module_output LIBRARY_OUTPUT_DIRECTORY)
  endif (WIN32)

  set_target_properties(${name}
    PROPERTIES
      DEFINE_SYMBOL    ${define}
      ${module_output} "${MODULE_OUTPUT_PATH}"
      PREFIX           "")

  install(
    TARGETS     ${name}
    DESTINATION "${MODULE_INSTALL_PATH}"
    COMPONENT   runtime)
endfunction (add_vistk_plugin name define)

add_subdirectory(processes)
