// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_PROCESSES_COMPUTE_STEREO_DEPTH_MAP_PROCESS_H
#define ARROWS_PROCESSES_COMPUTE_STEREO_DEPTH_MAP_PROCESS_H

#include <viame/pipeline_framework/process.h>

#include "viame_processes_measurement_export.h"

#include <viame/algorithm_framework/config/config_block.h>

namespace kwiver {

// ----------------------------------------------------------------
class VIAME_PROCESSES_MEASUREMENT_EXPORT compute_stereo_depth_map_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "compute_stereo_depth_map",
               "Compute a stereo depth map given two frames." )

  compute_stereo_depth_map_process( kwiver::vital::config_block_sptr const& config );
  virtual ~compute_stereo_depth_map_process();

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

#endif /* ARROWS_PROCESSES_COMPUTE_STEREO_DEPTH_MAP_PROCESS_H */
