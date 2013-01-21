project(vistk_python_vistk_image)

find_package(NumPy)

if (NUMPY_FOUND)
  include_directories(SYSTEM ${NUMPY_INCLUDE_DIR})

  if (NUMPY_VERSION VERSION_GREATER "1.7.0" OR
      NUMPY_VERSION VERSION_EQUAL "1.7.0")
    add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
  endif ()

  add_vistk_python_library(vil
    vistk/image
    vil.cxx)
  target_link_libraries(python-vistk.image-vil
    vil
    numpy_memory_chunk
    vistk_python_any_conversion
    vistk_python_util
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})
endif ()

make_python_init_file(vistk/image)
#  vil)
