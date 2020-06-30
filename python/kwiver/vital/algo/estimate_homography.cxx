// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <python/kwiver/vital/algo/trampoline/estimate_homography_trampoline.txx>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void estimate_homography(py::module &m)
{
  py::class_< kwiver::vital::algo::estimate_homography,
              std::shared_ptr<kwiver::vital::algo::estimate_homography>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::estimate_homography>,
              estimate_homography_trampoline<> >( m, "EstimateHomography" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::estimate_homography::static_type_name)
    .def("estimate",
         [](kwiver::vital::algo::estimate_homography const& self,
            kwiver::vital::feature_set_sptr feat1,
            kwiver::vital::feature_set_sptr feat2,
            kwiver::vital::match_set_sptr matches,
            double inlier_scale)
         {
           std::vector<bool> inliers;
           auto h = self.estimate(std::move(feat1), std::move(feat2), std::move(matches), inliers, inlier_scale);
           return h ? py::cast(std::make_pair(std::move(h), std::move(inliers)))
             : py::none();
         },
         py::arg("feat1"), py::arg("feat2"), py::arg("matches"),
         py::arg("inlier_scale") = 1.0)
    .def("estimate",
         [](kwiver::vital::algo::estimate_homography const& self,
            std::vector<kwiver::vital::vector_2d> const& pts1,
            std::vector<kwiver::vital::vector_2d> const& pts2,
            double inlier_scale)
         {
           std::vector<bool> inliers;
           auto h = self.estimate(pts1, pts2, inliers, inlier_scale);
           return h ? py::cast(std::make_pair(std::move(h), std::move(inliers)))
             : py::none();
         },
         py::arg("pts1"), py::arg("pts2"), py::arg("inlier_scale") = 1.0);
}
}
}
}
