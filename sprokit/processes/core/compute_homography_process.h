// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_COMPUTE_HOMOGRAPHY_PROCESS_H_
#define _KWIVER_COMPUTE_HOMOGRAPHY_PROCESS_H_

#include <sprokit/pipeline/process.h>
#include "kwiver_processes_export.h"

#include <memory>

namespace kwiver
{

// ----------------------------------------------------------------
class KWIVER_PROCESSES_NO_EXPORT compute_homography_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "compute_homography",
               "Compute a frame to frame homography based on tracks." )

  compute_homography_process( kwiver::vital::config_block_sptr const& config );
  virtual ~compute_homography_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class compute_homography_process

} // end namespace
#endif /* _KWIVER_COMPUTE_HOMOGRAPHY_PROCESS_H_ */
