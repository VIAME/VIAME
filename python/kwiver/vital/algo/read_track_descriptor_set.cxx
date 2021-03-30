// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/trampoline/read_track_descriptor_set_trampoline.txx>
#include <python/kwiver/vital/algo/read_track_descriptor_set.h>

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void read_track_descriptor_set(py::module &m)
{
  py::class_< kwiver::vital::algo::read_track_descriptor_set,
              std::shared_ptr<kwiver::vital::algo::read_track_descriptor_set>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::read_track_descriptor_set>,
              read_track_descriptor_set_trampoline<> >(m, "ReadTrackDescriptorSet")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::read_track_descriptor_set::static_type_name)
    .def("open", &kwiver::vital::algo::read_track_descriptor_set::open)
    .def("close", &kwiver::vital::algo::read_track_descriptor_set::close)
    .def("read_set", &kwiver::vital::algo::read_track_descriptor_set::read_set);
}
}
}
}
