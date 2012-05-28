include_directories("${vistk_source_dir}/src")
include_directories("${vistk_binary_dir}/src")

add_subdirectory(test)

add_subdirectory(pipeline)
add_subdirectory(pipeline_util)
