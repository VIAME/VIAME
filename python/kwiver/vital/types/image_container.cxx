// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/image_container.h>
#include <python/kwiver/vital/types/image_container.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace kwiver {
namespace vital  {
namespace python {
std::shared_ptr<s_image_cont_t>
kwiver::vital::python::image_container::new_cont(kwiver::vital::image &img)
{
  return std::shared_ptr<s_image_cont_t>(new s_image_cont_t(img));
}

kwiver::vital::image
kwiver::vital::python::image_container::get_image(std::shared_ptr<image_cont_t> self)
{
  kwiver::vital::image img;
  img.copy_from(self->get_image());
  return img;
}

void kwiver::vital::python::image_container::image_container(py::module& m)
{
  /*
   *
    Developer:
        python -c "import kwiver.vital.types; help(kwiver.vital.types.ImageContainer)"
        python -m xdoctest kwiver.vital.types ImageContainer --xdoc-force-dynamic
        python -m xdoctest kwiver.vital.types ImageContainer.asarray --xdoc-force-dynamic  # fix xdoctest to execute this
   *
   */
  py::class_<image_cont_t, std::shared_ptr<image_cont_t>>(m, "BaseImageContainer")
  .def("size", &image_cont_t::size)
  .def("width", &image_cont_t::width)
  .def("height", &image_cont_t::height)
  .def("depth", &image_cont_t::depth)
  .def("image", &kwiver::vital::python::image_container::get_image)
  .def("asarray",
    [](image_cont_t& img_cont)
    {
      py::object np_arr = kwiver::vital::python::image::asarray(img_cont.get_image());
      return np_arr;
    },
    py::doc(R"(
    Returns a copy of the internal image data as a numpy array.
    ")")
  )

  .def("__nice__", [](py::object& self) -> std::string {
    auto locals = py::dict(py::arg("self")=self);
    py::exec(R"(
      retval = 'whd={}x{}x{}'.format(self.width(), self.height(), self.depth())
      )", py::globals(), locals);
    return locals["retval"].cast<std::string>();
  })

  .def("__repr__", [](py::object& self) -> std::string {
    auto locals = py::dict(py::arg("self")=self);
    py::exec(R"(
      classname = self.__class__.__name__
      devnice = self.__nice__()
      retval = '<%s(%s) at %s>' % (classname, devnice, hex(id(self)))
      )", py::globals(), locals);
    return locals["retval"].cast<std::string>();
  })

  .def("__str__", [](py::object& self) -> std::string {
    auto locals = py::dict(py::arg("self")=self);
    py::exec(R"(
      classname = self.__class__.__name__
      devnice = self.__nice__()
      retval = '<%s(%s)>' % (classname, devnice)
      )", py::globals(), locals);
    return locals["retval"].cast<std::string>();
  });

  py::class_<s_image_cont_t, image_cont_t, std::shared_ptr<s_image_cont_t>>(m, "ImageContainer", R"(
    Example:
        >>> # Example using PIL utility
        >>> from kwiver.vital.types import ImageContainer
        >>> from PIL import Image as PILImage
        >>> from kwiver.vital.util import VitalPIL
        >>> import numpy as np
        >>> np_img = (np.random.rand(10, 20, 3) * 255).astype(np.uint8)
        >>> pil_img = PILImage.fromarray(np_img)
        >>> vital_img = VitalPIL.from_pil(pil_img)
        >>> self = ImageContainer(vital_img)
        >>> print(str(self))
        <ImageContainer(whd=20x10x3)>
        >>> vital_img2 = self.image()
        >>> pil_img2 = VitalPIL.get_pil_image(vital_img2)
        >>> np_img2 = np.asarray(pil_img2)
        >>> assert np.all(np_img2 == np_img)

    Example:
        >>> # Example using numpy conversion methdos
        >>> from kwiver.vital.types import ImageContainer
        >>> import numpy as np
        >>> np_img = (np.random.rand(10, 20, 3) * 255).astype(np.uint8)
        >>> self = ImageContainer.fromarray(np_img)
        >>> np_img2 = self.asarray()
        >>> assert np.all(np_img == np_img2)
    )")

  .def(py::init(&kwiver::vital::python::image_container::new_cont), py::arg("image"))

  // Create initializer based on numpy array type
  #define def_fromarray( T ) \
  .def_static("fromarray", \
              &kwiver::vital::python::image_container::new_image_container_from_numpy<T>, \
              py::arg("array"),\
      py::doc("Create an ImageContainer from a numpy array"))
  def_fromarray( uint8_t )
  def_fromarray( int8_t )
  def_fromarray( uint16_t )
  def_fromarray( int16_t )
  def_fromarray( uint32_t )
  def_fromarray( int32_t )
  def_fromarray( uint64_t )
  def_fromarray( int64_t )
  def_fromarray( float )
  def_fromarray( double )
  def_fromarray( bool );
  #undef def_fromarray
}
}}}
