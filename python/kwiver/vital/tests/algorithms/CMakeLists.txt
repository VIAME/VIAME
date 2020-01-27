set(__test_algorithms
  simple_image_object_detector
  )

set(test_algorithms
    __init__
    ${__test_algorithms}
   )

foreach (modname IN LISTS test_algorithms)
  kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/${modname}.py
    vital/tests/algorithms
    ${modname})
endforeach()
