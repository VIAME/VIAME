// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#pragma once

#include <sprokit/pipeline/process.h>

#include "kwiver_processes_export.h"

#include <memory>

namespace kwiver
{

// ----------------------------------------------------------------
/**
 * \class merge_images_process
 *
 * \brief Merges two images
 *
 * \iports
 * \iport{image}
 *
 * \oports
 * \oport{image1}
 * \oport{image2}
 *
 */
class KWIVER_PROCESSES_NO_EXPORT merge_images_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "merge_image",
               "Merge multiple images into one.\n\n"
               "This process merges all the channels in two input "
               "images into a single output image based on the "
               "implementation "
               "of the 'merge_images' algorithm selected.\n"
               "This process has two input ports accepting "
               "the images. The first two connect commands will be "
               "accepted as the input ports. The actual input port "
               "names do not matter.\n"
               "The channels from the image from the first port "
               "connected will be added to the output image first."
    )

  merge_images_process( kwiver::vital::config_block_sptr const& config );
  virtual ~merge_images_process();

protected:
  void _configure() override;
  void _step() override;
  void input_port_undefined(port_t const& port) override;

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class merge_images_process

} // end namespace
