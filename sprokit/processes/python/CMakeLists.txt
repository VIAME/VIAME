#
# KWIVER processes
#

set(kwiver_python_subdir "python${PYTHON_VERSION}${PYTHON_ABIFLAGS}")
set(kwiver_python_output_path "${KWIVER_BINARY_DIR}/lib/${kwiver_python_subdir}")

kwiver_create_python_plugin_init(kwiver)
kwiver_create_python_plugin_init(kwiver/processes)

# add python processes here
kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/kw_print_number_process.py
  kwiver/processes
  kw_print_number_process)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/homography_writer.py
  kwiver/processes
  homography_writer)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/ProcessImage.py
  kwiver/processes
  ProcessImage)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/ApplyDescriptor.py
  kwiver/processes
  ApplyDescriptor)
