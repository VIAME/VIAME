// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_TRANSPORT_FILE_TRANSPORT_SEND_PROCESS_H
#define KWIVER_TRANSPORT_FILE_TRANSPORT_SEND_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "kwiver_processes_transport_export.h"

namespace kwiver {

// ----------------------------------------------------------------
/**
 * \class file_transport_send_process
 *
 * \brief Writes serialized data to a file.
 *
 *
 * \oport{message}
 */
class KWIVER_PROCESSES_TRANSPORT_NO_EXPORT file_transport_send_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "file_transport_send",
               "Writes the serialized buffer to a file." )

  file_transport_send_process( kwiver::vital::config_block_sptr const& config );
  virtual ~file_transport_send_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class file_transport_send_process

}  // end namespace

#endif // KWIVER_TRANSPORT_FILE_TRANSPORT_SEND_PROCESS_H
