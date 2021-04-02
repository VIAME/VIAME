// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/trampoline/split_image_trampoline.txx>
#include <python/kwiver/vital/algo/split_image.h>

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void split_image(py::module &m)
{
  py::class_< kwiver::vital::algo::split_image,
              std::shared_ptr<kwiver::vital::algo::split_image>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::split_image>,
              split_image_trampoline<> >(m, "SplitImage")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::split_image::static_type_name)
    .def("split", &kwiver::vital::algo::split_image::split);
}
}
}
}
