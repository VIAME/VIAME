// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/match_descriptor_sets.h>
#include "algorithm_python.txx"
#include "match_descriptor_sets_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void match_descriptor_sets(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::match_descriptor_sets,
               std::shared_ptr<kwiver::vital::algo::match_descriptor_sets>,
               kwiver::vital::algorithm,
               match_descriptor_sets_trampoline<> > instance(m,  "MatchDescriptorSets");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::match_descriptor_sets::interface_name)
    .def("append_to_index", &kwiver::vital::algo::match_descriptor_sets::append_to_index, py::doc(R"( Add a descriptor set to the inverted file system.

 Add a descriptor set and frame number to the inverted file system.
 Future matching results may include this frame in their results.

 \param[in] desc   set of descriptors associated with this frame
 \param[in] frame  frame number indexing the descriptors
 \returns None)"), py::arg("desc"), py::arg("frame"))
    .def("query", &kwiver::vital::algo::match_descriptor_sets::query, py::doc(R"( Query the inverted file system for similar sets of descriptors.

 Query the inverted file system and return the frames containing the most
 similar sets descriptors.

 \param[in] desc  set of descriptors to match
 \returns vector of possibly matching frames found by the query)"), py::arg("desc"))
    .def("query_and_append", &kwiver::vital::algo::match_descriptor_sets::query_and_append, py::doc(R"( Query the inverted file system and append the descriptors.

 This method is equivalent to calling query() followed by
 append_to_index();
 however, depending on the implementation, it may be faster to call this
 single function when both operations are required.

 \param[in] desc   set of descriptors to match and append
 \param[in] frame  frame number indexing the descriptors
 \returns vector of possibly matching frames found by the query)"), py::arg("desc"), py::arg("frame"))
    ;
  register_algorithm< kwiver::vital::algo::match_descriptor_sets > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
