// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_TRACK_OBJECTS_PROCESS_H_
#define _KWIVER_TRACK_OBJECTS_PROCESS_H_

#include <viame/pipeline_framework/process.h>
#include "viame_processes_object_trackers_export.h"

#include <memory>

namespace kwiver
{

// ----------------------------------------------------------------
/**
 * \class track_objects_process
 *
 * \brief Tracks detected objects across frames using a configurable
 *        track_objects algorithm implementation.
 *
 * \iports
 * \iport{timestamp} Frame timestamp
 * \iport{image} Input image
 * \iport{detected_object_set} Detected objects to track
 *
 * \oports
 * \oport{object_track_set} Tracked objects
 *
 */
class VIAME_PROCESSES_OBJECT_TRACKERS_EXPORT track_objects_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "track_objects",
               "Tracks detected objects across frames.")

  typedef sprokit::process base_t;

  track_objects_process( kwiver::vital::config_block_sptr const& config );
  virtual ~track_objects_process();

protected:
    virtual void _configure();
    virtual void _step();

private:
    void make_ports();
    void make_config();

    class priv;
    const std::unique_ptr<priv> d;
 }; // end class track_objects_process

} // end namespace
#endif /* _KWIVER_TRACK_OBJECTS_PROCESS_H_ */
