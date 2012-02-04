project(vistk_python_image)

find_package(NumPy)

if (NUMPY_FOUND)
  include_directories(${NUMPY_INCLUDE_DIR})

  add_vistk_python_library(vil
    vistk/image
    vil.cxx)
  target_link_libraries(python-vil
    vil
    numpy_memory_chunk
    vistk_python_any_conversion
    ${Boost_PYTHON_LIBRARY}
    ${PYTHON_LIBRARIES})
endif (NUMPY_FOUND)

make_python_init_file(vistk/image)
#  vil)
