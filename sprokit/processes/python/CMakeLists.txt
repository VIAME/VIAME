#
# KWIVER processes
#

kwiver_create_python_init(kwiver)
kwiver_create_python_init(kwiver/processes)

# add python processes here
kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/kw_print_number_process.py
  kwiver/processes
  kw_print_number_process)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/homography_writer.py
  kwiver/processes
  homography_writer)

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/simple_homog_tracker.py
  kwiver/processes
  simple_homog_tracker)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/ProcessImage.py
  kwiver/processes
  ProcessImage)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/ApplyDescriptor.py
  kwiver/processes
  ApplyDescriptor)
