// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/trampoline/merge_images_trampoline.txx>
#include <python/kwiver/vital/algo/merge_images.h>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void merge_images(py::module &m)
{
  py::class_< kwiver::vital::algo::merge_images,
              std::shared_ptr<kwiver::vital::algo::merge_images>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::merge_images>,
              merge_images_trampoline<> >(m, "MergeImages")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::merge_images::static_type_name)
    .def("match",
        &kwiver::vital::algo::merge_images::merge);
}
}
}
}
