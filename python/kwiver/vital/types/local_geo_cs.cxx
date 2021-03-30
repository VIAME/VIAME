// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/local_geo_cs.h>

#include <pybind11/pybind11.h>

#include <memory>

namespace py = pybind11;
namespace kv = kwiver::vital;

PYBIND11_MODULE( local_geo_cs, m )
{
  py::class_< kv::local_geo_cs, std::shared_ptr< kv::local_geo_cs > >( m, "LocalGeoCS" )
  .def( py::init<>() )
  .def_property( "geo_origin", &kv::local_geo_cs::origin, &kv::local_geo_cs::set_origin )
  ;
  m.def( "read_local_geo_cs_from_file",      &kv::read_local_geo_cs_from_file );
  m.def( "write_local_geo_cs_to_file",       &kv::write_local_geo_cs_to_file, py::return_value_policy::reference );
}
