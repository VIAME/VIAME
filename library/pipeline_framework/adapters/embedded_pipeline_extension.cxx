// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "embedded_pipeline_extension.h"
#include <viame/algorithm_framework/vital_config.h>

namespace viame {

embedded_pipeline_extension::
embedded_pipeline_extension()
{ }

// ----------------------------------------------------------------------------
void
embedded_pipeline_extension::
configure( [[maybe_unused]] viame::config_block_sptr const conf )
{ }

// ----------------------------------------------------------------------------
viame::config_block_sptr
embedded_pipeline_extension::
get_configuration() const
{
  auto conf = viame::config_block::empty_config();
  return conf;
}

} // namespace viame
