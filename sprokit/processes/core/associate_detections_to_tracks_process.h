// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_ASSOCIATE_DETECTIONS_TO_TRACKS_PROCESS_H_
#define _KWIVER_ASSOCIATE_DETECTIONS_TO_TRACKS_PROCESS_H_

#include "kwiver_processes_export.h"

#include <sprokit/pipeline/process.h>

#include <memory>

namespace kwiver
{

// -----------------------------------------------------------------------------
/**
 * \class associate_detections_to_tracks_process
 *
 * \brief Associates new object detections to existing tracks.
 *
 * \iports
 * \iport{timestamp}
 * \iport{image}
 * \iport{tracks}
 * \iport{detections}
 * \iport{matrix_d}
 *
 * \oports
 * \oport{tracks}
 * \oport{unused_detections}
 * \oport{all_detections}
 */
class KWIVER_PROCESSES_NO_EXPORT associate_detections_to_tracks_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "associate_detections_to_tracks",
               "Associates new detections to existing tracks given a cost matrix." )

  associate_detections_to_tracks_process( vital::config_block_sptr const& config );
  virtual ~associate_detections_to_tracks_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class associate_detections_to_tracks_process

} // end namespace
#endif /* _KWIVER_ASSOCIATE_DETECTIONS_TO_TRACKS_PROCESS_H_ */
