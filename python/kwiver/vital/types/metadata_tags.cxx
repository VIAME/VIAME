// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/metadata_tags.h>
#include <viame/core_types/metadata_traits.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace kv = kwiver::vital;

PYBIND11_MODULE( metadata_tags, m )
{
  auto tags_enum = py::enum_< kv::vital_metadata_tag >( m, "tags" );
  for( auto tag = kv::VITAL_META_UNKNOWN; tag < kv::VITAL_META_LAST_TAG;
       tag = static_cast< kv::vital_metadata_tag >(
         static_cast< size_t >( tag ) + 1 ) )
  {
    auto const enum_name =
      std::string( "VITAL_META_" ) + kv::tag_traits_by_tag( tag ).enum_name();
    tags_enum.value( enum_name.c_str(), tag );
  }

  tags_enum.value( "VITAL_META_LAST_TAG", kv::VITAL_META_LAST_TAG );
}
