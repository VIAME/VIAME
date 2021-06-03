// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <python/kwiver/vital/algo/trampoline/estimate_fundamental_matrix_trampoline.txx>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void estimate_fundamental_matrix(py::module &m)
{
  py::class_< kwiver::vital::algo::estimate_fundamental_matrix,
              std::shared_ptr<kwiver::vital::algo::estimate_fundamental_matrix>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::estimate_fundamental_matrix>,
              estimate_fundamental_matrix_trampoline<> >( m, "EstimateFundamentalMatrix" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::estimate_fundamental_matrix::static_type_name)
    .def("estimate",
         ( kwiver::vital::fundamental_matrix_sptr
           (kwiver::vital::algo::estimate_fundamental_matrix::* )
           ( const kwiver::vital::feature_set_sptr,
             const kwiver::vital::feature_set_sptr,
             const kwiver::vital::match_set_sptr,
             std::vector< bool >&,
             double ) const )
         &kwiver::vital::algo::estimate_fundamental_matrix::estimate )
    .def("estimate",
         ( kwiver::vital::fundamental_matrix_sptr
           (kwiver::vital::algo::estimate_fundamental_matrix::* )
           ( const std::vector< kwiver::vital::vector_2d >&,
             const std::vector< kwiver::vital::vector_2d >&,
             std::vector< bool >&,
             double ) const )
         &kwiver::vital::algo::estimate_fundamental_matrix::estimate );
}
}
}
}
