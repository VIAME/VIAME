// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for process instrumentation.
 */

#include "process_instrumentation.h"

#include <viame/algorithm_framework/vital_config.h>
#include <viame/pipeline_framework/process.h>

namespace sprokit {

process_instrumentation::
process_instrumentation()
  : m_process( nullptr )
{ }

void
process_instrumentation::
set_process( sprokit::process const& proc )
{
  m_process = &proc;
}

void
process_instrumentation::
configure( [[maybe_unused]] kwiver::vital::config_block_sptr const config )
{ }

kwiver::vital::config_block_sptr
process_instrumentation::
get_configuration() const
{
  auto conf = kwiver::vital::config_block::empty_config();
  return conf;
}

std::string
process_instrumentation::
process_name() const
{
  return m_process->name ();
}

} // end namespace sprokit
