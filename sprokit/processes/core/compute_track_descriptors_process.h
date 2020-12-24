// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_COMPUTE_TRACK_DESCRIPTORS_PROCESS_H_
#define _KWIVER_COMPUTE_TRACK_DESCRIPTORS_PROCESS_H_

#include "kwiver_processes_export.h"

#include <sprokit/pipeline/process.h>

#include <vital/types/track_descriptor_set.h>

#include <memory>

namespace kwiver
{

// -----------------------------------------------------------------------------
/**
 * \class compute_track_descriptors_process
 *
 * \brief Computes track descriptors along object tracks or object detections.
 *
 * \iports
 * \iport{timestamp}
 * \iport{image}
 * \iport{tracks}
 * \iport{detections}
 *
 * \oports
 * \oport{track_descriptor_set}
 */
class KWIVER_PROCESSES_NO_EXPORT compute_track_descriptors_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "compute_track_descriptors",
               "Compute track descriptors on the input tracks or detections." )

  compute_track_descriptors_process( vital::config_block_sptr const& config );
  virtual ~compute_track_descriptors_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  void push_outputs( vital::track_descriptor_set_sptr& to_output );

  class priv;
  const std::unique_ptr<priv> d;
}; // end class compute_track_descriptors_process

} // end namespace
#endif /* _KWIVER_COMPUTE_TRACK_DESCRIPTORS_PROCESS_H_ */
