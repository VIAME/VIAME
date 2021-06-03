// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/trampoline/match_descriptor_sets_trampoline.txx>
#include <python/kwiver/vital/algo/match_descriptor_sets.h>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void match_descriptor_sets(py::module &m)
{
  py::class_< kwiver::vital::algo::match_descriptor_sets,
              std::shared_ptr<kwiver::vital::algo::match_descriptor_sets>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::match_descriptor_sets>,
              match_descriptor_sets_trampoline<> >(m, "MatchDescriptorSets")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::match_descriptor_sets::static_type_name)
    .def("query",
        &kwiver::vital::algo::match_descriptor_sets::query)
    .def("append_to_index",
        &kwiver::vital::algo::match_descriptor_sets::append_to_index)
    .def("query_and_append",
        &kwiver::vital::algo::match_descriptor_sets::query_and_append);
}
}
}
}
