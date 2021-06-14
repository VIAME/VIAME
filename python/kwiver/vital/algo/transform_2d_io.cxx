// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/trampoline/transform_2d_io_trampoline.txx>
#include <python/kwiver/vital/algo/transform_2d_io.h>

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void transform_2d_io(py::module &m)
{
  py::class_< kwiver::vital::algo::transform_2d_io,
              std::shared_ptr<kwiver::vital::algo::transform_2d_io>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::transform_2d_io>,
              transform_2d_io_trampoline<> >(m, "Transform2DIO")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::transform_2d_io::static_type_name)
    .def("save",
         &kwiver::vital::algo::transform_2d_io::save)
    .def("load",
         &kwiver::vital::algo::transform_2d_io::load);
}
}
}
}
