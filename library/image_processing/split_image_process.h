// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_SPLIT_IMAGE_PROCESS_H_
#define KWIVER_SPLIT_IMAGE_PROCESS_H_

#include <viame/pipeline_framework/process.h>

#include "viame_processes_image_processing_export.h"

#include <memory>

namespace kwiver
{

// ----------------------------------------------------------------
/**
 * \class split_image_process
 *
 * \brief Splits an image into multiple smaller images.
 *
 * \iports
 * \iport{image}
 *
 * \oports
 * \oport{image1}
 * \oport{image2}
 *
 */
class VIAME_PROCESSES_IMAGE_PROCESSING_EXPORT split_image_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "split_image",
               "Split a image into multiple smaller images." )

  split_image_process( kwiver::vital::config_block_sptr const& config );
  virtual ~split_image_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class split_image_process

} // end namespace

#endif
