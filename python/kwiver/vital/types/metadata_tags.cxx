// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/metadata_tags.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace kv = kwiver::vital;

#define REGISTER_TAG_ENUM( TAG, NAME, T, ... ) \
.value( "VITAL_META_" #TAG, kv::vital_metadata_tag::VITAL_META_ ## TAG )

PYBIND11_MODULE( metadata_tags, m )
{
  py::enum_< kv::vital_metadata_tag >( m, "tags" )
    KWIVER_VITAL_METADATA_TAGS( REGISTER_TAG_ENUM )
    .value( "VITAL_META_LAST_TAG", kv::VITAL_META_LAST_TAG )
  ;
}

#undef REGISTER_TAG_ENUM
