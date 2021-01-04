// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "embedded_pipeline_extension.h"
#include <vital/vital_config.h>

namespace kwiver {

embedded_pipeline_extension::
embedded_pipeline_extension()
{ }

// ----------------------------------------------------------------------------
void
embedded_pipeline_extension::
configure( VITAL_UNUSED kwiver::vital::config_block_sptr const conf )
{ }

// ----------------------------------------------------------------------------
kwiver::vital::config_block_sptr
embedded_pipeline_extension::
get_configuration() const
{
  auto conf = kwiver::vital::config_block::empty_config();
  return conf;
}

} // end namespace kwiver
