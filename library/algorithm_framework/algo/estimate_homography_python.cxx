// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/estimate_homography.h>
#include "algorithm_python.txx"
#include "estimate_homography_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void estimate_homography(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<viame::algo::estimate_homography,
               std::shared_ptr<viame::algo::estimate_homography>,
               viame::algorithm,
               estimate_homography_trampoline<> > instance(m,  "EstimateHomography");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::estimate_homography::interface_name)
    .def("estimate", (viame::homography_sptr (viame::algo::estimate_homography::*)(::viame::feature_set_sptr, ::viame::feature_set_sptr, ::viame::match_set_sptr, ::std::vector<bool> &, double) const) &viame::algo::estimate_homography::estimate, py::doc(R"( Estimate a homography matrix from corresponding features

 If estimation fails, a NULL-containing sptr is returned

 \param [in]  feat1 the set of all features from the source image
 \param [in]  feat2 the set of all features from the destination image
 \param [in]  matches the set of correspondences between \a feat1 and \a
 feat2
 \param [out] inliers for each match in \a matcher, the value is true if
                      this pair is an inlier to the homography estimate
 \param [in]  inlier_scale error distance tolerated for matches to be
 inliers)"), py::arg("feat1"), py::arg("feat2"), py::arg("matches"), py::arg("inliers"), py::arg("inlier_scale") = 1.)
    .def("estimate", (viame::homography_sptr (viame::algo::estimate_homography::*)(::std::vector<viame::vector_<2, double> > const &, ::std::vector<viame::vector_<2, double> > const &, ::std::vector<bool> &, double) const) &viame::algo::estimate_homography::estimate, py::doc(R"( Estimate a homography matrix from corresponding points

 If estimation fails, a NULL-containing sptr is returned

 \param [in]  pts1 the vector or corresponding points from the source image
 \param [in]  pts2 the vector of corresponding points from the destination
 image
 \param [out] inliers for each point pair, the value is true if
                      this pair is an inlier to the homography estimate
 \param [in]  inlier_scale error distance tolerated for matches to be
 inliers)"), py::arg("pts1"), py::arg("pts2"), py::arg("inliers"), py::arg("inlier_scale") = 1.)
    ;
  register_algorithm< viame::algo::estimate_homography > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
