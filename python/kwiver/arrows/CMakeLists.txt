add_subdirectory(serialize)
add_subdirectory(core)

if(KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                         arrows
                         __init__)
kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/simple_image_object_detector.py
                         arrows
                         simple_image_object_detector)
