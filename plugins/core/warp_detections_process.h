/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Warp detections using a 2D transform loaded from a file
 */

#ifndef VIAME_CORE_WARP_DETECTIONS_PROCESS_H
#define VIAME_CORE_WARP_DETECTIONS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "viame_processes_core_export.h"

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Warp detection bounding boxes into another camera's image space
 *
 * The transform is loaded once at configure time via a transform_2d_io
 * reader (default "auto": DIVE camera registration .json or plain text
 * 3x3 homography). Each detection's box corners are mapped through the
 * transform and re-boxed axis-aligned.
 */
class VIAME_PROCESSES_CORE_NO_EXPORT warp_detections_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  warp_detections_process( kwiver::vital::config_block_sptr const& config );
  virtual ~warp_detections_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class warp_detections_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_WARP_DETECTIONS_PROCESS_H
