// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/close_loops.h>
#include "algorithm_python.txx"
#include "close_loops_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void close_loops(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::close_loops,
               std::shared_ptr<kwiver::vital::algo::close_loops>,
               kwiver::vital::algorithm,
               close_loops_trampoline<> > instance(m,  "CloseLoops");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::close_loops::interface_name)
    .def("stitch", &kwiver::vital::algo::close_loops::stitch, py::doc(R"( Attempt to perform closure operation and stitch tracks together.

 \param frame_number the frame number of the current frame
 \param input the input feature track set to stitch
 \param image image data for the current frame
 \param mask Optional mask image where positive values indicate
                  regions to consider in the input image.
 \returns an updated set of feature tracks after the stitching operation)"), py::arg("frame_number"), py::arg("input"), py::arg("image"), py::arg("mask") = py::none())
    ;
  register_algorithm< kwiver::vital::algo::close_loops > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
