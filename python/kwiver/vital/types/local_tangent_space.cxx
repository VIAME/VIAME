// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/local_tangent_space.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <memory>

namespace py = pybind11;
namespace kv = kwiver::vital;

PYBIND11_MODULE( local_tangent_space, m )
{
  py::class_<
    kv::local_tangent_space,
    std::shared_ptr< kv::local_tangent_space > >(
    m,
    "LocalTangentSpace" )
    .def( py::init<>() )
    .def( py::init< kv::geo_point const& >(), py::arg( "origin" ) )
    .def_property_readonly( "origin", &kv::local_tangent_space::origin )
    .def( "valid", &kv::local_tangent_space::valid )
    .def(
      "to_local",
      py::overload_cast< kv::geo_point const& >(
        &kv::local_tangent_space::to_local, py::const_ ),
      py::arg( "global_point" ) )
    .def(
      "to_local",
      py::overload_cast< kv::rotation_d const&, kv::geo_point const& >(
        &kv::local_tangent_space::to_local, py::const_ ),
      py::arg( "global_rotation" ), py::arg( "global_point" ) )
    .def(
      "to_global",
      py::overload_cast< kv::vector_3d const& >(
        &kv::local_tangent_space::to_global, py::const_ ),
      py::arg( "local_point" ) )
    .def(
      "to_global",
      py::overload_cast< kv::rotation_d const&, kv::geo_point const& >(
        &kv::local_tangent_space::to_global, py::const_ ),
      py::arg( "local_rotation" ), py::arg( "global_point" ) )
  ;
  m.def(
    "read_local_tangent_space_from_file",
    &kv::read_local_tangent_space_from_file );
  m.def(
    "write_local_tangent_space_to_file",
    &kv::write_local_tangent_space_to_file );
}
