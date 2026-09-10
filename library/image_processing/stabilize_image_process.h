// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_STABILIZE_IMAGE_PROCESS_H_
#define _KWIVER_STABILIZE_IMAGE_PROCESS_H_

#include <viame/pipeline_framework/process.h>
#include "viame_processes_image_processing_export.h"

#include <memory>

namespace kwiver
{

// ----------------------------------------------------------------
/**
 * \class stabilize_image_process
 *
 * \brief Stabilizes a series of image.
 *
 * \iports
 * \iport{timestamp}
 * \iport{image}
 *
 * \oports
 * \oport{src_to_ref_homography}
 *
 */
class VIAME_PROCESSES_IMAGE_PROCESSING_EXPORT stabilize_image_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "stabilize_image",
               "Generate current-to-reference image homographies." )

  stabilize_image_process( kwiver::vital::config_block_sptr const& config );
  virtual ~stabilize_image_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class stabilize_image_process

} // end namespace

#endif /* _KWIVER_STABILIZE_IMAGE_PROCESS_H_ */
