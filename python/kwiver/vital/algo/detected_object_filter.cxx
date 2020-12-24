// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/detected_object_filter_trampoline.txx>
#include <python/kwiver/vital/algo/detected_object_filter.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {
void detected_object_filter(py::module &m)
{
  py::class_< kwiver::vital::algo::detected_object_filter,
              std::shared_ptr<kwiver::vital::algo::detected_object_filter>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::detected_object_filter>,
              detected_object_filter_trampoline<> >( m, "DetectedObjectFilter" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::detected_object_filter::static_type_name)
    .def("filter",
         &kwiver::vital::algo::detected_object_filter::filter);
}
}
}
}
