project(vistk_tests)

set(vistk_test_data_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/data)

if (VISTK_ENABLE_PYTHON)
  add_subdirectory(python)
endif (VISTK_ENABLE_PYTHON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(vistk)
