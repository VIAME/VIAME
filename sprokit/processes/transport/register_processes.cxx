// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "kwiver_processes_transport_export.h"

#include <sprokit/pipeline/process_factory.h>
#include <vital/plugin_loader/plugin_loader.h>

// -- list processes to register --
#include "file_transport_send_process.h"

#if WITH_ZMQ
#include "zmq_transport_send_process.h"
#include "zmq_transport_receive_process.h"
#endif

// ---------------------------------------------------------------------------------------
/** \brief Regsiter processes
 *
 *
 */
extern "C"
KWIVER_PROCESSES_TRANSPORT_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  using namespace sprokit;

  process_registrar reg( vpm, "kwiver_processes_transport_export" );

  if ( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_process< kwiver::file_transport_send_process >();

#if WITH_ZMQ

  reg.register_process< kwiver::zmq_transport_send_process >();
  reg.register_process< kwiver::zmq_transport_receive_process >();

#endif

 // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  reg.mark_module_as_loaded();
} // register_processes
