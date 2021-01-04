// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <utility>

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/detected_object_set_input_trampoline.txx>
#include <python/kwiver/vital/algo/detected_object_set_input.h>

namespace py = pybind11;

using dosi = kwiver::vital::algo::detected_object_set_input;
namespace kwiver {
namespace vital {
namespace python {
void detected_object_set_input(py::module &m)
{
  py::class_< dosi,
              std::shared_ptr<dosi>,
              kwiver::vital::algorithm_def<dosi>,
              detected_object_set_input_trampoline<> >(m, "DetectedObjectSetInput")
    .def(py::init())
    .def_static("static_type_name", &dosi::static_type_name)
    .def("read_set",
	 [](dosi& self) {
	   std::pair<kwiver::vital::detected_object_set_sptr, std::string> result;
	   bool has_result = self.read_set(result.first, result.second);
	   return has_result ? py::cast(result) : py::cast(nullptr);
	 },
	 R"(Return a pair of the next DetectedObjectSet and the corresponding
file name, or None if the input is exhausted)")
    .def("open", &dosi::open)
    .def("close", &dosi::close)
    ;
}
}
}
}
