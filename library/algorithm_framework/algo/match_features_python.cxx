// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/match_features.h>
#include "algorithm_python.txx"
#include "match_features_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void match_features(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<viame::algo::match_features,
               std::shared_ptr<viame::algo::match_features>,
               viame::algorithm,
               match_features_trampoline<> > instance(m,  "MatchFeatures");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::match_features::interface_name)
    .def("match", &viame::algo::match_features::match, py::doc(R"( Match one set of features and corresponding descriptors to another

 \param feat1 the first set of features to match
 \param desc1 the descriptors corresponding to \a feat1
 \param feat2 the second set fof features to match
 \param desc2 the descriptors corresponding to \a feat2
 \returns a set of matching indices from \a feat1 to \a feat2)"), py::arg("feat1"), py::arg("desc1"), py::arg("feat2"), py::arg("desc2"))
    ;
  register_algorithm< viame::algo::match_features > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
