/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Warp an image using a 2D homography loaded from a file
 */

#ifndef VIAME_CORE_WARP_IMAGE_PROCESS_H
#define VIAME_CORE_WARP_IMAGE_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "viame_processes_core_export.h"

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Warp an image into another camera's image space
 *
 * The transform is loaded once at configure time via a transform_2d_io
 * reader and must be a homography (DIVE camera registration .json or plain
 * text 3x3 homography). The transform maps this image's coordinates into
 * the target camera's, the same convention as warp_detections; the output
 * size defaults to the input's and can follow another camera's via the
 * optional size_image port.
 */
class VIAME_PROCESSES_CORE_NO_EXPORT warp_image_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  warp_image_process( kwiver::vital::config_block_sptr const& config );
  virtual ~warp_image_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class warp_image_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_WARP_IMAGE_PROCESS_H
