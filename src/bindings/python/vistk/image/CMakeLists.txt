project(vistk_python_vistk_image)

find_package(NumPy)

if (NUMPY_FOUND)
  include_directories(SYSTEM ${NUMPY_INCLUDE_DIR})

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
