// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Canonical KPF types.
 */

#include "kpf_canonical_types.h"
#include <arrows/kpf/yaml/kpf_packet.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( "arrows.kpf.kpf_canonical_types" ) );

using std::string;

namespace kwiver {
namespace vital {
namespace kpf {

namespace canonical
{

kv_t
::kv_t( const string& k, const string& v )
  : key(k), val(v)
{
  if ( str2style( k ) != packet_style::INVALID )
  {
    LOG_ERROR( main_logger, "KV packet '" << k << "' / '" << v << "'; key value is already a KPF packet type-- file won't parse" );
    // throw an error?
  }
}

} // ...canonical
} // ...kpf
} // ...vital
} // ...kwiver
