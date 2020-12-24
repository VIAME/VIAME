// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/compute_stereo_depth_map_trampoline.txx>
#include <python/kwiver/vital/algo/compute_stereo_depth_map.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {
void compute_stereo_depth_map(py::module &m)
{
  py::class_< kwiver::vital::algo::compute_stereo_depth_map,
              std::shared_ptr<kwiver::vital::algo::compute_stereo_depth_map>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::compute_stereo_depth_map>,
              compute_stereo_depth_map_trampoline<> >( m, "ComputeStereoDepthMap" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::compute_stereo_depth_map::static_type_name)
    .def("compute",
         &kwiver::vital::algo::compute_stereo_depth_map::compute);
}
}
}
}
