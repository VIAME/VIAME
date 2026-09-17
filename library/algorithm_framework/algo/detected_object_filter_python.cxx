// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/detected_object_filter.h>
#include "algorithm_python.txx"
#include "detected_object_filter_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void detected_object_filter(py::module& m)
{
  py::module::import("viame.config");
  py::module::import("viame.types");

    py::class_<viame::algo::detected_object_filter,
               std::shared_ptr<viame::algo::detected_object_filter>,
               viame::algorithm,
               detected_object_filter_trampoline<> > instance(m,  "DetectedObjectFilter");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::detected_object_filter::interface_name)
    .def("filter", &viame::algo::detected_object_filter::filter, py::doc(R"( Filter set of detected objects.

 This method applies a filter to the input set to create an output
 set. The input set of detections is unmodified.

 \param input_set Set of detections to be filtered.
 \returns Filtered set of detections.)"), py::arg("input_set"))
    ;
  register_algorithm< viame::algo::detected_object_filter > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
