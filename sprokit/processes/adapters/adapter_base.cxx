// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for adapter base class.
 */

#include "adapter_base.h"

#include <vital/util/bounded_buffer.h>

#include <iterator>

namespace kwiver {
namespace adapter {

//----------------------------------------------------------------
adapter_base
::adapter_base()
  : m_interface_queue( new kwiver::vital::bounded_buffer< kwiver::adapter::adapter_data_set_t > (2) )
{
}

adapter_base
::~adapter_base()
{
}

// ------------------------------------------------------------------
kwiver::adapter::interface_ref_t
adapter_base
::get_interface_queue()
{
  return m_interface_queue;
}

// ------------------------------------------------------------------
sprokit::process::ports_t
adapter_base
::port_list() const
{
  sprokit::process::ports_t ports;

  // return a copy of our port names
  std::copy( m_active_ports.begin(), m_active_ports.end(), std::back_inserter( ports ) );

  return ports;
}

} } // end namespace kwiver
