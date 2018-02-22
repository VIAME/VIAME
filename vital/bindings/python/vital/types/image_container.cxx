/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <vital/types/image_container.h>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

typedef kwiver::vital::image_container image_cont;
typedef kwiver::vital::simple_image_container s_image_cont;

// We need to return a shared pointer--otherwise, pybind11 may lose the subtype
std::shared_ptr<s_image_cont>
new_cont(kwiver::vital::image &img)
{
  return std::shared_ptr<s_image_cont>(new s_image_cont(img));
}

// We need to do a deep copy instead of just calling get_image, so we can ref track in python
kwiver::vital::image
get_image(std::shared_ptr<image_cont> self)
{
  kwiver::vital::image img;
  img.copy_from(self->get_image());
  return img;
}

PYBIND11_MODULE(image_container, m)
{
  /*
   *

    Developer:
        python -c "import vital.types; help(vital.types.ImageContainer)"
        python -m xdoctest vital.types ImageContainer --xdoc-dynamic
        python -m xdoctest vital.types ImageContainer.asarray --xdoc-dynamic  # fix xdoctest to execute this

   *
   */
  py::class_<image_cont, std::shared_ptr<image_cont>>(m, "BaseImageContainer")
  .def("size", &image_cont::size)
  .def("width", &image_cont::width)
  .def("height", &image_cont::height)
  .def("depth", &image_cont::depth)
  .def("image", &get_image)

  .def("asarray",
    [](image_cont& self) -> py::array
    {
      auto locals = py::dict(py::arg("self")=self);
      py::exec(R"(
        from vital.util import VitalPIL
        import numpy as np
        vital_img = self.image()
        pil_img = VitalPIL.get_pil_image(vital_img)
        retval = np.asarray(pil_img)
        )", py::globals(), locals);
      return locals["retval"].cast<py::array>();
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
  })

  ;

  py::class_<s_image_cont, image_cont, std::shared_ptr<s_image_cont>>(m, "ImageContainer", R"(
    Example:
        >>> # Example using PIL utility
        >>> from vital.types import ImageContainer
        >>> from PIL import Image as PILImage
        >>> from vital.util import VitalPIL
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
        >>> from vital.types import ImageContainer
        >>> import numpy as np
        >>> np_img = (np.random.rand(10, 20, 3) * 255).astype(np.uint8)
        >>> self = ImageContainer.fromarray(np_img)
        >>> np_img2 = self.asarray()
        >>> assert np.all(np_img == np_img2)
    )")

  .def(py::init(&new_cont), py::arg("image"))

  .def_static("fromarray",
    [](py::array& arr) -> s_image_cont {
      auto locals = py::dict(py::arg("arr")=arr);
      py::exec(R"(
          from vital.types import ImageContainer
          from PIL import Image as PILImage
          from vital.util import VitalPIL
          pil_img = PILImage.fromarray(arr)
          vital_img = VitalPIL.from_pil(pil_img)
          self = ImageContainer(vital_img)
          )", py::globals(), locals);
      return locals["self"].cast<s_image_cont>();
    }, py::doc(R"(
    Create an ImageContainer from a numpy array

    Example:
        >>> from vital.types import ImageContainer
        >>> import numpy as np
        >>> np_img = (np.random.rand(10, 20, 3) * 255).astype(np.uint8)
        >>> self = ImageContainer.fromarray(np_img)
        >>> print(str(self))
        <ImageContainer(whd=20x10x3)>
        >>> np_img2 = self.asarray()
        >>> assert np.all(np_img == np_img2)
    )")
  )
  ;
}
