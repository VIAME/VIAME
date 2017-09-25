#
# KWIVER processes
#

kwiver_create_python_plugin_init(kwiver)
kwiver_create_python_plugin_init(kwiver/processes)

# add python processes here
if( FALSE )
  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/kw_print_number_process.py
    kwiver/processes
    kw_print_number_process)
endif()

if( FALSE )
  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/homography_writer.py
    kwiver/processes
    homography_writer)
endif()

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/ProcessImage.py
  kwiver/processes
  ProcessImage)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/ApplyDescriptor.py
  kwiver/processes
  ApplyDescriptor)
