// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Unwrap the object detections from object tracks.

#ifndef _KWIVER_UNWRAP_DETECTIONS_PROCESS_H
#define _KWIVER_UNWRAP_DETECTIONS_PROCESS_H

#include "viame_processes_object_trackers_export.h"
#include <viame/pipeline_framework/process.h>

#include <memory>

namespace kwiver {

// -------------------------------------------------------------------------------
class VIAME_PROCESSES_OBJECT_TRACKERS_EXPORT unwrap_detections_process
  : public sprokit::process
{
public:
  PLUGIN_INFO(
    "unwrap_detections",
    "Unwrap object detections from object tracks." )

  // -- CONSTRUCTORS --
  unwrap_detections_process( kwiver::vital::config_block_sptr const& config );
  virtual ~unwrap_detections_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;

  const std::unique_ptr< priv > d;
}; // end class unwrap_detections_process

} // end namespace

#endif /* _KWIVER_UNWRAP_DETECTIONS_PROCESS_H */
