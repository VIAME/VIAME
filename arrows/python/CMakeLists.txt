
set(kwiver_python_subdir "python${PYTHON_VERSION}${PYTHON_ABIFLAGS}")
set(kwiver_python_output_path "${KWIVER_BINARY_DIR}/lib/${kwiver_python_subdir}")

kwiver_create_python_init(kwiver/arrows/python)
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/simple_image_detector.py
                        kwiver/arrows/python
                        simple_image_detector)

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/detected_object_set_input_coco.py
                        kwiver/arrows/python
                        detected_object_set_input_coco)

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/detected_object_set_output_coco.py
                        kwiver/arrows/python
                        detected_object_set_output_coco)
