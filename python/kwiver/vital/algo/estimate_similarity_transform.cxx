// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/estimate_similarity_transform.h>
#include <python/kwiver/vital/algo/trampoline/estimate_similarity_transform_trampoline.txx>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void estimate_similarity_transform(py::module &m)
{
  py::class_< kwiver::vital::algo::estimate_similarity_transform,
              std::shared_ptr<kwiver::vital::algo::estimate_similarity_transform>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::estimate_similarity_transform>,
              estimate_similarity_transform_trampoline<> >( m, "EstimateSimilarityTransform" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::estimate_similarity_transform::static_type_name)
    .def("estimate_transform",
         ( kwiver::vital::similarity_d
           ( kwiver::vital::algo::estimate_similarity_transform::* )
           ( std::vector< kwiver::vital::vector_3d > const&,
             std::vector< kwiver::vital::vector_3d > const& ) const )
       &kwiver::vital::algo::estimate_similarity_transform::estimate_transform )
    .def("estimate_transform",
         ( kwiver::vital::similarity_d
           ( kwiver::vital::algo::estimate_similarity_transform::* )
           ( std::vector< kwiver::vital::camera_perspective_sptr > const&,
             std::vector< kwiver::vital::camera_perspective_sptr > const& ) const )
       &kwiver::vital::algo::estimate_similarity_transform::estimate_transform )
    .def("estimate_transform",
         ( kwiver::vital::similarity_d
           ( kwiver::vital::algo::estimate_similarity_transform::* )
           ( std::vector< kwiver::vital::landmark_sptr > const&,
             std::vector< kwiver::vital::landmark_sptr > const& ) const )
       &kwiver::vital::algo::estimate_similarity_transform::estimate_transform )
    .def("estimate_transform",
         ( kwiver::vital::similarity_d
           ( kwiver::vital::algo::estimate_similarity_transform::* )
           ( kwiver::vital::camera_map_sptr const,
             kwiver::vital::camera_map_sptr const ) const )
       &kwiver::vital::algo::estimate_similarity_transform::estimate_transform )
    .def("estimate_transform",
         ( kwiver::vital::similarity_d
           ( kwiver::vital::algo::estimate_similarity_transform::* )
           ( kwiver::vital::landmark_map_sptr const,
             kwiver::vital::landmark_map_sptr const ) const )
       &kwiver::vital::algo::estimate_similarity_transform::estimate_transform );

}
}
}
}
