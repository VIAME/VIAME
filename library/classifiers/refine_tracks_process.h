// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_PROCESSES_REFINE_TRACKS_PROCESS_H
#define ARROWS_PROCESSES_REFINE_TRACKS_PROCESS_H

#include <viame/pipeline_framework/process.h>
#include "viame_processes_classifiers_export.h"
#include <viame/algorithm_framework/config/config_block.h>

namespace kwiver {

/// @brief Track refiner process.
///
/// \process Refines object tracks for a given frame.
///
/// This process uses the refine_tracks algorithm to improve track quality
/// on a per-frame basis. It can re-segment masks, filter tracks, adjust
/// bounding boxes, and add new objects based on query criteria.
///
/// \iports
/// \iport{image} Image for the current frame
/// \iport{timestamp} Timestamp for the current frame
/// \iport{object_track_set} Object tracks to refine
///
/// \oports
/// \oport{object_track_set} Refined object tracks
///
class VIAME_PROCESSES_CLASSIFIERS_EXPORT refine_tracks_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "refine_tracks",
               "Refines object tracks for a given frame" )

  refine_tracks_process( kwiver::vital::config_block_sptr const& config );
  virtual ~refine_tracks_process();

protected:
  virtual void _configure();
  virtual void _step();
  virtual void _finalize();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
};

} // end namespace kwiver

#endif // ARROWS_PROCESSES_REFINE_TRACKS_PROCESS_H
