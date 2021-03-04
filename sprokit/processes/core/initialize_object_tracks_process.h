// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_INITIALIZE_OBJECT_TRACKS_PROCESS_H_
#define _KWIVER_INITIALIZE_OBJECT_TRACKS_PROCESS_H_

#include "kwiver_processes_export.h"

#include <sprokit/pipeline/process.h>

#include <memory>

namespace kwiver
{

// -----------------------------------------------------------------------------
/**
 * \class initialize_object_tracks_process
 *
 * \brief Initialized new tracks given object detections.
 *
 *  On optional input track port will union the input track set and newly
 *  initialized tracks.
 *
 * \iports
 * \iport{timestamp}
 * \iport{image}
 * \iport{detections}
 * \iport{tracks}
 *
 * \oports
 * \oport{tracks}
 */
class KWIVER_PROCESSES_NO_EXPORT initialize_object_tracks_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "initialize_object_tracks",
               "Initialize new object tracks given detections for the current frame." )

  initialize_object_tracks_process( vital::config_block_sptr const& config );
  virtual ~initialize_object_tracks_process();

protected:
  virtual void _configure();
  virtual void _step();
  virtual void _init();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class initialize_object_tracks_process

} // end namespace
#endif /* _KWIVER_INITIALIZE_OBJECT_TRACKS_PROCESS_H_ */
