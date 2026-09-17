// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/estimate_fundamental_matrix.h>
#include "algorithm_python.txx"
#include "estimate_fundamental_matrix_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void estimate_fundamental_matrix(py::module& m)
{
  py::module::import("viame.config");
  py::module::import("viame.types");

    py::class_<viame::algo::estimate_fundamental_matrix,
               std::shared_ptr<viame::algo::estimate_fundamental_matrix>,
               viame::algorithm,
               estimate_fundamental_matrix_trampoline<> > instance(m,  "EstimateFundamentalMatrix");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::estimate_fundamental_matrix::interface_name)
    .def("estimate", (viame::fundamental_matrix_sptr (viame::algo::estimate_fundamental_matrix::*)(::viame::feature_set_sptr const, ::viame::feature_set_sptr const, ::viame::match_set_sptr const, ::std::vector<bool> &, double) const) &viame::algo::estimate_fundamental_matrix::estimate, py::doc(R"( Estimate an fundamental matrix from corresponding features

 \param [in]  feat1 the set of all features from the first image
 \param [in]  feat2 the set of all features from the second image
 \param [in]  matches the set of correspondences between \a feat1 and
                      \a feat2
 \param [out] inliers for each point pair, the value is true if
                      this pair is an inlier to the estimate
 \param [in]  inlier_scale error distance tolerated for matches to be
 inliers)"), py::arg("feat1"), py::arg("feat2"), py::arg("matches"), py::arg("inliers"), py::arg("inlier_scale") = 1.)
    .def("estimate", (viame::fundamental_matrix_sptr (viame::algo::estimate_fundamental_matrix::*)(::std::vector<viame::vector_<2, double> > const &, ::std::vector<viame::vector_<2, double> > const &, ::std::vector<bool> &, double) const) &viame::algo::estimate_fundamental_matrix::estimate, py::doc(R"( Estimate an fundamental matrix from corresponding points

 \param [in]  pts1 the vector or corresponding points from the first image
 \param [in]  pts2 the vector of corresponding points from the second image
 \param [out] inliers for each point pair, the value is true if
                      this pair is an inlier to the estimate
 \param [in]  inlier_scale error distance tolerated for matches to be
 inliers)"), py::arg("pts1"), py::arg("pts2"), py::arg("inliers"), py::arg("inlier_scale") = 1.)
    ;
  register_algorithm< viame::algo::estimate_fundamental_matrix > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
