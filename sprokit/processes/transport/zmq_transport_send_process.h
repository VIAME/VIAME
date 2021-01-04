// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_TRANSPORT_ZMQ_TRANSPORT_SEND_PROCESS_H
#define KWIVER_TRANSPORT_ZMQ_TRANSPORT_SEND_PROCESS_H

#include <sprokit/pipeline/process.h>
#include <zmq.hpp>

#include "kwiver_processes_transport_export.h"

namespace kwiver {

// ----------------------------------------------------------------
/*
 * zmq_transport_send_process
 *
 *  Writes serialized data to a file.
 */
class KWIVER_PROCESSES_TRANSPORT_NO_EXPORT zmq_transport_send_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "zmq_transport_send",
               "Send serialized buffer to ZMQ transport." )

  zmq_transport_send_process( kwiver::vital::config_block_sptr const& config );
  virtual ~zmq_transport_send_process();

protected:
  virtual void _configure();
  virtual void _init();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class zmq_transport_send_process

}  // end namespace

#endif // KWIVER_TRANSPORT_ZMQ_TRANSPORT_SEND_PROCESS_H
