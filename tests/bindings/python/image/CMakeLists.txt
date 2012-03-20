project(vistk_test_python_image)

find_package(NumPy)

if (NUMPY_FOUND)
  ##############################
  # vil tests
  ##############################
  vistk_build_python_test(vil test-vil.py)

  vistk_make_python_test(vil import)
  vistk_make_python_test(vil vil_to_numpy)
  vistk_make_python_test(vil numpy_to_vil)
  vistk_make_python_test(vil datum)
  vistk_make_python_test(vil memory)
endif (NUMPY_FOUND)
