// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/trampoline/uuid_factory_trampoline.txx>
#include <python/kwiver/vital/algo/uuid_factory.h>

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void uuid_factory(py::module &m)
{
  py::class_< kwiver::vital::algo::uuid_factory,
              std::shared_ptr<kwiver::vital::algo::uuid_factory>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::uuid_factory>,
              uuid_factory_trampoline<> >(m, "UUIDFactory")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::uuid_factory::static_type_name)
    .def("create_uuid",
         &kwiver::vital::algo::uuid_factory::create_uuid);
}
}
}
}
