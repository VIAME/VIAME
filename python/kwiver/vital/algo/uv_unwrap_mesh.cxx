// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/trampoline/uv_unwrap_mesh_trampoline.txx>
#include <python/kwiver/vital/algo/uv_unwrap_mesh.h>

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void uv_unwrap_mesh(py::module &m)
{
  py::class_< kwiver::vital::algo::uv_unwrap_mesh,
              std::shared_ptr<kwiver::vital::algo::uv_unwrap_mesh>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::uv_unwrap_mesh>,
              uv_unwrap_mesh_trampoline<> >(m, "UVUnwrapMesh")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::uv_unwrap_mesh::static_type_name)
    .def("unwrap",
         &kwiver::vital::algo::uv_unwrap_mesh::unwrap);
}
}
}
}
