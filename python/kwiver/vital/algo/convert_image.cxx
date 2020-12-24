// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/convert_image_trampoline.txx>
#include <python/kwiver/vital/algo/convert_image.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {
void convert_image(py::module &m)
{
  py::class_< kwiver::vital::algo::convert_image,
              std::shared_ptr<kwiver::vital::algo::convert_image>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::convert_image>,
              convert_image_trampoline<> >( m, "ConvertImage" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::convert_image::static_type_name)
    .def("convert",
         &kwiver::vital::algo::convert_image::convert);
}
}
}
}
