// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/merge_detections.h>
#include "algorithm_python.txx"
#include "merge_detections_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void merge_detections(py::module& m)
{
  py::module::import("viame.config");
  py::module::import("viame.types");

    py::class_<viame::algo::merge_detections,
               std::shared_ptr<viame::algo::merge_detections>,
               viame::algorithm,
               merge_detections_trampoline<> > instance(m,  "MergeDetections");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::merge_detections::interface_name)
    .def("merge", &viame::algo::merge_detections::merge, py::doc(R"( Merge several detection sets into one

 Combines the supplied detection sets into a single set. Implementations
 decide how overlapping detections are resolved, for example by
 non-maximum suppression or by fusing their type scores.

 \param sets Detection sets to merge. All sets are expected to describe
             the same frame.

 \returns The merged detection set. Empty if \p sets is empty or contains
          no detections.)"), py::arg("sets"))
    ;
  register_algorithm< viame::algo::merge_detections > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
