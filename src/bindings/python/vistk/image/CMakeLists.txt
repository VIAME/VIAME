project(vistk_python_vistk_image)

find_package(NumPy)

if (NUMPY_FOUND)
  add_vistk_python_library(vil
    vistk/image
    vil.cxx)
  target_link_libraries(python-vistk.image-vil
    LINK_PRIVATE
      vil
      vistk_python_any_conversion
      vistk_python_numpy
      vistk_python_util
      ${${boost_python_library_var}}
      ${PYTHON_LIBRARIES})
endif ()

make_python_init_file(vistk/image)
#  vil)
