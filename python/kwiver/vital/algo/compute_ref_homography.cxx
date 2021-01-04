// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/compute_ref_homography_trampoline.txx>
#include <python/kwiver/vital/algo/compute_ref_homography.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {
void compute_ref_homography(py::module &m)
{
  py::class_< kwiver::vital::algo::compute_ref_homography,
              std::shared_ptr<kwiver::vital::algo::compute_ref_homography>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::compute_ref_homography>,
              compute_ref_homography_trampoline<> >( m, "ComputeRefHomography" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::compute_ref_homography::static_type_name)
    .def("estimate",
         &kwiver::vital::algo::compute_ref_homography::estimate);
}
}
}
}
