// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>

#include <python/kwiver/vital/algo/trampoline/compute_track_descriptors_trampoline.txx>
#include <python/kwiver/vital/algo/compute_track_descriptors.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {
void compute_track_descriptors(py::module &m)
{
  py::class_< kwiver::vital::algo::compute_track_descriptors,
              std::shared_ptr<kwiver::vital::algo::compute_track_descriptors>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::compute_track_descriptors>,
              compute_track_descriptors_trampoline<> >( m, "ComputeTrackDescriptors" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::compute_track_descriptors::static_type_name)
    .def("compute",
         &kwiver::vital::algo::compute_track_descriptors::compute)
    .def("flush",
         &kwiver::vital::algo::compute_track_descriptors::flush);
}
}
}
}
