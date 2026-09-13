viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/kwiver_process.py
  sprokit/processes
  kwiver_process )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/apply_descriptor.py
  sprokit/processes
  apply_descriptor )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/homography_writer.py
  sprokit/processes
  homography_writer )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/kw_print_number_process.py
  sprokit/processes
  kw_print_number_process )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/process_image.py
  sprokit/processes
  process_image )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                          sprokit/processes
                          __init__ )
