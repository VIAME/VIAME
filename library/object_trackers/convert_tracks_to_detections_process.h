// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_CONVERT_TRACKS_TO_DETECTIONS_PROCESS_H_
#define _KWIVER_CONVERT_TRACKS_TO_DETECTIONS_PROCESS_H_

#include "viame_processes_object_trackers_export.h"

#include <viame/pipeline_framework/process.h>

#include <memory>

namespace kwiver {

// -----------------------------------------------------------------------------

/**
 * \class convert_tracks_to_detections_process
 *
 * \brief Computes an object track set to a detection set for the given frame
 *
 * \iports
 * \iport{timestamp}
 * \iport{object_track_set}
 *
 * \oports
 * \oport{detection_set}
 */
class VIAME_PROCESSES_OBJECT_TRACKERS_EXPORT convert_tracks_to_detections_process
  : public sprokit::process
{
public:
  PLUGIN_INFO(
    "convert_tracks_to_detections",
    "Convert input object track sets into detection sets for each frame." )

  convert_tracks_to_detections_process(
    vital::config_block_sptr const& config );
  virtual ~convert_tracks_to_detections_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;


  const std::unique_ptr< priv > d;
}; // end class convert_tracks_to_detections_process

} // end namespace

#endif /* _KWIVER_CONVERT_TRACKS_TO_DETECTIONS_PROCESS_H_ */
