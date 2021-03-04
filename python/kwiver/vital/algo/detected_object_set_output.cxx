// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/detected_object_set_output_trampoline.txx>
#include <python/kwiver/vital/algo/detected_object_set_output.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {
using doso = kwiver::vital::algo::detected_object_set_output;

class doso_publicist
: public doso
{
public:
  using doso::doso;
  using doso::filename;
};

void detected_object_set_output(py::module &m)
{
  py::class_< doso,
              std::shared_ptr<doso>,
              kwiver::vital::algorithm_def<doso>,
              detected_object_set_output_trampoline<> >(m, "DetectedObjectSetOutput")
    .def(py::init())
    .def_static("static_type_name", &doso::static_type_name)
    .def("write_set", &doso::write_set)
    .def("complete", &doso::complete)
    .def("open", &doso::open)
    .def("close", &doso::close)
    .def("filename", &doso_publicist::filename)
    ;
}
}
}
}
