
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/kwiver_process.py
  sprokit/processes
  kwiver_process )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/ApplyDescriptor.py
  sprokit/processes
  ApplyDescriptor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/homography_writer.py
  sprokit/processes
  homography_writer )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/kw_print_number_process.py
  sprokit/processes
  kw_print_number_process )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/ProcessImage.py
  sprokit/processes
  ProcessImage )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                          sprokit/processes
                          __init__ )
