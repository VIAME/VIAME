/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <vital/types/metadata_map.h>

#include <python/kwiver/vital/util/pybind11.h>

#include <pybind11/pybind11.h>

#include <memory>

namespace py = pybind11;
namespace kv = kwiver::vital;

class metadata_map_trampoline
  : public kv::metadata_map
{
public:
  using metadata_map::metadata_map;

  size_t size() const override;
  kv::metadata_map::map_metadata_t metadata() const override;
  bool has_item( kv::vital_metadata_tag tag, kv::frame_id_t fid ) const override;
  kv::metadata_item const& get_item( kv::vital_metadata_tag tag, kv::frame_id_t fid ) const override;
  kv::metadata_vector get_vector( kv::frame_id_t fid ) const override;
  std::set< kv::frame_id_t > frames() override;
};


PYBIND11_MODULE( metadata_map, m )
{
  py::class_< kv::metadata_map,
              std::shared_ptr< kv::metadata_map >,
              metadata_map_trampoline >( m, "MetadataMap" )
  .def( py::init<>() )
  .def( "size",       &kv::metadata_map::size )
  .def( "metadata",   &kv::metadata_map::metadata )
  .def( "has_item",   &kv::metadata_map::has_item )
  .def( "get_item",   &kv::metadata_map::get_item )
  .def( "get_vector", &kv::metadata_map::get_vector )
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
              kv::metadata_map >( m, "SimpleMetadataMap" );
  // Everything will be inherited from metadata_map
}

// Now the trampoline's overrides
size_t
metadata_map_trampoline
::size() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    size_t,
    kv::metadata_map,
    size,
  );
}
kv::metadata_map::map_metadata_t
metadata_map_trampoline
::metadata() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    kv::metadata_map::map_metadata_t,
    kv::metadata_map,
    metadata,

  );
}
bool
metadata_map_trampoline
::has_item( kv::vital_metadata_tag tag, kv::frame_id_t fid ) const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
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
  VITAL_PYBIND11_OVERLOAD_PURE(
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
  VITAL_PYBIND11_OVERLOAD_PURE(
    kv::metadata_vector,
    kv::metadata_map,
    get_vector,
    fid
  );
}

std::set< kv::frame_id_t >
metadata_map_trampoline
::frames()
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    std::set< kv::frame_id_t >,
    kv::metadata_map,
    frames,
  );
}
