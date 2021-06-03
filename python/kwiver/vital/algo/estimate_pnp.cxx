// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/estimate_pnp_trampoline.txx>
#include <python/kwiver/vital/algo/estimate_pnp.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void estimate_pnp(py::module &m)
{
  py::class_< kwiver::vital::algo::estimate_pnp,
              std::shared_ptr<kwiver::vital::algo::estimate_pnp>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::estimate_pnp>,
              estimate_pnp_trampoline<> >( m, "EstimatePNP" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::estimate_pnp::static_type_name)
    .def("estimate",
         &kwiver::vital::algo::estimate_pnp::estimate);
}
}
}
}
