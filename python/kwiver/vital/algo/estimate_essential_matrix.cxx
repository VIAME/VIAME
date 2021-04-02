// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <python/kwiver/vital/algo/trampoline/estimate_essential_matrix_trampoline.txx>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void estimate_essential_matrix(py::module &m)
{
  py::class_< kwiver::vital::algo::estimate_essential_matrix,
              std::shared_ptr<kwiver::vital::algo::estimate_essential_matrix>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::estimate_essential_matrix>,
              estimate_essential_matrix_trampoline<> >( m, "EstimateEssentialMatrix" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::estimate_essential_matrix::static_type_name)
    .def("estimate",
         ( kwiver::vital::essential_matrix_sptr
          (kwiver::vital::algo::estimate_essential_matrix::* )
          ( const kwiver::vital::feature_set_sptr,
            const kwiver::vital::feature_set_sptr,
            const kwiver::vital::match_set_sptr,
            const kwiver::vital::camera_intrinsics_sptr,
            const kwiver::vital::camera_intrinsics_sptr,
            std::vector< bool >&,
            double ) const )
         &kwiver::vital::algo::estimate_essential_matrix::estimate )
    .def("estimate",
         ( kwiver::vital::essential_matrix_sptr
           (kwiver::vital::algo::estimate_essential_matrix::* )
           ( const kwiver::vital::feature_set_sptr,
             const kwiver::vital::feature_set_sptr,
             const kwiver::vital::match_set_sptr,
             const kwiver::vital::camera_intrinsics_sptr,
             std::vector< bool >&,
             double ) const )
         &kwiver::vital::algo::estimate_essential_matrix::estimate )
    .def("estimate",
         ( kwiver::vital::essential_matrix_sptr
           (kwiver::vital::algo::estimate_essential_matrix::* )
           ( const std::vector< kwiver::vital::vector_2d >&,
             const std::vector< kwiver::vital::vector_2d >&,
             const kwiver::vital::camera_intrinsics_sptr,
             std::vector< bool >&,
             double ) const )
         &kwiver::vital::algo::estimate_essential_matrix::estimate )
    .def("estimate",
         ( kwiver::vital::essential_matrix_sptr
           ( kwiver::vital::algo::estimate_essential_matrix::* )
           ( const std::vector< kwiver::vital::vector_2d >&,
             const std::vector< kwiver::vital::vector_2d >&,
             const kwiver::vital::camera_intrinsics_sptr,
             const kwiver::vital::camera_intrinsics_sptr,
             std::vector< bool >&,
             double ) const )
         &kwiver::vital::algo::estimate_essential_matrix::estimate );
}
}
}
}
