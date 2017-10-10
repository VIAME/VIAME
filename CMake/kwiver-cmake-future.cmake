# -----------------------------------------------------------------------------
function(kwiver_import_cmake_future BASE_PATH)
  message(STATUS "Import CMake future from '${BASE_PATH}'")
  file(GLOB _cmake_future_versions RELATIVE ${BASE_PATH} "${BASE_PATH}/*/")
  foreach(_version ${_cmake_future_versions})
    message(STATUS "Import CMake future '${_version}'")
    if(IS_DIRECTORY "${BASE_PATH}/${_version}")
      if(CMAKE_VERSION VERSION_LESS ${_version})
        list(APPEND CMAKE_MODULE_PATH "${BASE_PATH}/${_version}")
      endif()
    endif()
  endforeach()

  set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
endfunction()

###############################################################################

message(STATUS "Current path: '${CMAKE_MODULE_PATH}'")
kwiver_import_cmake_future(${CMAKE_CURRENT_LIST_DIR}/future)
