// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/trampoline/keyframe_selection_trampoline.txx>
#include <python/kwiver/vital/algo/keyframe_selection.h>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void keyframe_selection(py::module &m)
{
  py::class_< kwiver::vital::algo::keyframe_selection,
              std::shared_ptr<kwiver::vital::algo::keyframe_selection>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::keyframe_selection>,
              keyframe_selection_trampoline<> >(m, "KeyframeSelection")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::keyframe_selection::static_type_name)
    .def("select",
        &kwiver::vital::algo::keyframe_selection::select);
}
}
}
}
