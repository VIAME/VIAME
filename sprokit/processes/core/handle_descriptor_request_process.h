// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_HANDLE_DESCRIPTOR_REQUEST_PROCESS_H_
#define _KWIVER_HANDLE_DESCRIPTOR_REQUEST_PROCESS_H_

#include "kwiver_processes_export.h"

#include <sprokit/pipeline/process.h>

#include <memory>

namespace kwiver
{

// -----------------------------------------------------------------------------
/**
 * \class handle_descriptor_request_process
 *
 * \brief Generates association matrix between old tracks and new detections
 *        for use in object tracking.
 *
 * \iports
 * \iport{descriptor_request}
 *
 * \oports
 * \oport{track_descriptor_set}
 * \oport{image_container}
 * \oport{filename}
 * \oport{stream_id}
 */
class KWIVER_PROCESSES_NO_EXPORT handle_descriptor_request_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "handle_descriptor_request",
               "Handle a new descriptor request, producing desired "
               "descriptors on the input." )

  handle_descriptor_request_process( vital::config_block_sptr const& config );
  virtual ~handle_descriptor_request_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class handle_descriptor_request_process

} // end namespace
#endif /* _KWIVER_HANDLE_DESCRIPTOR_REQUEST_PROCESS_H_ */
