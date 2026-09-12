// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/metadata_map.h>

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace kv = kwiver::vital;

class metadata_map_trampoline
  : public kv::metadata_map
{
public:
  using metadata_map::metadata_map;

  size_t size() const override;
  kv::metadata_map::map_metadata_t metadata() const override;
  bool has_item(
    kv::vital_metadata_tag tag,
    kv::frame_id_t fid ) const override;
  kv::metadata_item const& get_item(
    kv::vital_metadata_tag tag,
    kv::frame_id_t fid ) const override;
  kv::metadata_vector get_vector( kv::frame_id_t fid ) const override;
  std::set< kv::frame_id_t > frames() const override;
};

PYBIND11_MODULE( metadata_map, m )
{
  py::class_< kv::metadata_map,
    std::shared_ptr< kv::metadata_map >,
    metadata_map_trampoline >( m, "MetadataMap" )
    .def( py::init<>() )
    .def( "size",       &kv::metadata_map::size )
    .def( "metadata",   &kv::metadata_map::metadata )
    .def(
      "has_item",
      &kv::metadata_map::has_item, py::arg( "tag" ), py::arg( "frame_id" ) )
    .def(
      "get_item",
      &kv::metadata_map::get_item,
      py::return_value_policy::reference, py::arg( "tag" ),
      py::arg( "frame_id" ) )
    .def( "get_vector", &kv::metadata_map::get_vector, py::arg( "frame_id" ) )
    .def( "frames",     &kv::metadata_map::frames )
  // Note that we are skipping the templated has and get methods.
  // Those methods are templated over the vital_metadata_tag enums
  // which have over 100 values. We would have to instantiate them all here
  // because of how pybind deals with templates. Those methods simply call
  // the non templated version (has_item or get_item, respectively), which are
  // bound, so no functionality is lost.
  ;

  py::class_< kv::simple_metadata_map,
    std::shared_ptr< kv::simple_metadata_map >,
    kv::metadata_map >( m, "SimpleMetadataMap" )
    .def( py::init<>() )
    .def( py::init< kv::metadata_map::map_metadata_t >() );
  // Everything will be inherited from metadata_map
}

// Now the trampoline's overrides
size_t
metadata_map_trampoline
::size() const
{
  PYBIND11_OVERLOAD_PURE(
    size_t,
    kv::metadata_map,
    size,
  );
}

kv::metadata_map::map_metadata_t
metadata_map_trampoline
::metadata() const
{
  PYBIND11_OVERLOAD_PURE(
    kv::metadata_map::map_metadata_t,
    kv::metadata_map,
    metadata,

  );
}

bool
metadata_map_trampoline
::has_item( kv::vital_metadata_tag tag, kv::frame_id_t fid ) const
{
  PYBIND11_OVERLOAD_PURE(
    bool,
    kv::metadata_map,
    has_item,
    tag,
    fid
  );
}

kv::metadata_item const&
metadata_map_trampoline
::get_item( kv::vital_metadata_tag tag, kv::frame_id_t fid ) const
{
  PYBIND11_OVERLOAD_PURE(
    kv::metadata_item const&,
    kv::metadata_map,
    get_item,
    tag,
    fid
  );
}

kv::metadata_vector
metadata_map_trampoline
::get_vector( kv::frame_id_t fid ) const
{
  PYBIND11_OVERLOAD_PURE(
    kv::metadata_vector,
    kv::metadata_map,
    get_vector,
    fid
  );
}

std::set< kv::frame_id_t >
metadata_map_trampoline
::frames() const
{
  PYBIND11_OVERLOAD_PURE(
    std::set< kv::frame_id_t >,
    kv::metadata_map,
    frames,
  );
}
