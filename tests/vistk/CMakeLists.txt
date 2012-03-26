include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_BINARY_DIR}/src)

add_subdirectory(test)

add_subdirectory(pipeline)
add_subdirectory(pipeline_util)
