if( KWIVER_ENABLE_PYTORCH )
  add_subdirectory( pytorch )
endif()

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/kwiver_process.py
  sprokit/processes
  kwiver_process )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/apply_descriptor.py
  sprokit/processes
  apply_descriptor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/homography_writer.py
  sprokit/processes
  homography_writer )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/simple_homog_tracker.py
  sprokit/processes
  simple_homog_tracker)

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/kw_print_number_process.py
  sprokit/processes
  kw_print_number_process )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/process_image.py
  sprokit/processes
  process_image )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                          sprokit/processes
                          __init__ )
