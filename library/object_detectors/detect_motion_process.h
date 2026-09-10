// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_DETECT_MOTION_PROCESS_H
#define SPROKIT_PROCESSES_DETECT_MOTION_PROCESS_H

#include "viame_processes_object_detectors_export.h"
#include <viame/pipeline_framework/process.h>
#include <viame/algorithm_framework/config/config_block.h>

namespace kwiver {

// ----------------------------------------------------------------
/**
 * @brief Motion detection process.
 *
 */
class VIAME_PROCESSES_OBJECT_DETECTORS_EXPORT detect_motion_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "detect_motion",
               "Detect motion in a sequence of images" )

  detect_motion_process( kwiver::vital::config_block_sptr const& config );
  virtual ~detect_motion_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
};

} // end namespace

#endif /* SPROKIT_PROCESSES_DETECT_MOTION_PROCESS_H */
