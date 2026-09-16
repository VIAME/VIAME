// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/compute_stereo_depth_map.h>
#include "algorithm_python.txx"
#include "compute_stereo_depth_map_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void compute_stereo_depth_map(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<viame::algo::compute_stereo_depth_map,
               std::shared_ptr<viame::algo::compute_stereo_depth_map>,
               viame::algorithm,
               compute_stereo_depth_map_trampoline<> > instance(m,  "ComputeStereoDepthMap");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::compute_stereo_depth_map::interface_name)
    .def("compute", &viame::algo::compute_stereo_depth_map::compute, py::doc(R"( Compute a stereo depth map given two images

 \throws image_size_mismatch_exception
    When the given input image sizes do not match.

 \param left_image contains the first image to process
 \param right_image contains the second image to process
 \returns a depth map image)"), py::arg("left_image"), py::arg("right_image"))
    ;
  register_algorithm< viame::algo::compute_stereo_depth_map > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
