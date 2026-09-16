// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/filter_features.h>
#include "algorithm_python.txx"
#include "filter_features_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void filter_features(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<viame::algo::filter_features,
               std::shared_ptr<viame::algo::filter_features>,
               viame::algorithm,
               filter_features_trampoline<> > instance(m,  "FilterFeatures");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::filter_features::interface_name)
    .def("filter", (viame::feature_set_sptr (viame::algo::filter_features::*)(::viame::feature_set_sptr) const) &viame::algo::filter_features::filter, py::doc(R"( Filter a feature set and return a subset of the features

 The default implementation call the pure virtual function
 filter(feature_set_sptr feat, std::vector<size_t> &indices) const
 \param [in] input The feature set to filter
 \returns a filtered version of the feature set (simple_feature_set))"), py::arg("input"))
    .def("filter", (viame::algo::filter_features::filter_return_value (viame::algo::filter_features::*)(::viame::feature_set_sptr, ::viame::descriptor_set_sptr) const) &viame::algo::filter_features::filter, py::arg("feat"), py::arg("descr"))
    ;
  register_algorithm< viame::algo::filter_features > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
