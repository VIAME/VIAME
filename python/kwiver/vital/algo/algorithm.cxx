// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <vital/algo/algorithm.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <python/kwiver/vital/algo/algorithm.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {
void algorithm(py::module &m)
{
  py::class_<kwiver::vital::algorithm, std::shared_ptr<kwiver::vital::algorithm>,
             algorithm_trampoline<>>(m, "_algorithm")
    .def_property("impl_name", &kwiver::vital::algorithm::impl_name, &kwiver::vital::algorithm::set_impl_name)
    .def("get_configuration", &kwiver::vital::algorithm::get_configuration)
    .def("set_configuration", &kwiver::vital::algorithm::set_configuration)
    .def("check_configuration", &kwiver::vital::algorithm::check_configuration);
}
}
}
}
