// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/geo_MGRS.h>

#include <pybind11/pybind11.h>

#include <memory>
#include <sstream>

namespace py=pybind11;
namespace kv=kwiver::vital;

PYBIND11_MODULE( geo_MGRS, m )
{
  py::class_< kv::geo_MGRS, std::shared_ptr< kv::geo_MGRS > >( m, "GeoMGRS" )
  .def( py::init<>() )
  .def( py::init<const std::string&>() )
  .def( "is_empty", &kv::geo_MGRS::is_empty )
  .def( "is_valid", &kv::geo_MGRS::is_valid )
  .def( "set_coord", &kv::geo_MGRS::set_coord )
  .def( "coord", &kv::geo_MGRS::coord )
  .def( "__eq__", [] ( kv::geo_MGRS const& self, kv::geo_MGRS const& other )
  {
    return self == other;
  })
  .def( "__ne__", [] ( kv::geo_MGRS const& self, kv::geo_MGRS const& other )
  {
    return self != other;
  })
  .def( "__str__", [] ( kv::geo_MGRS const& self )
  {
    std::stringstream str;
    str << self;
    return str.str();
  })
  ;
}
