/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef _VIAME_DETECT_IN_SUBREGIONS_PROCESS_H
#define _VIAME_DETECT_IN_SUBREGIONS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <vital/config/config_block.h>

#include "viame_processes_opencv_export.h"

namespace viame
{

// -----------------------------------------------------------------------------
/**
 * @brief Detect in subregions process.
 *
 * This process is intended to be used after an initial object detection step
 * identifies potentially interesting regions within an image, defined by a
 * detected_object_set, to run a subsequent, potentially more
 * expensive, detector/classifier algorithm. Each bounding box from the input
 * detected_object_set defines a sub-image, which is passed to the specified
 * detection algorithm. The result is a new, presumably more accurate,
 * detected_object_set, which may not have direct correlation with the input set
 * of image_object_detection other than being contained within the union of its
 * bounding boxes.
 *
 */
class VIAME_PROCESSES_OPENCV_NO_EXPORT detect_in_subregions_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "detect_in_subregions",
               "Run a detection algorithm on all of the chips represented "
               "by an incoming detected_object_set" )

  detect_in_subregions_process( kwiver::vital::config_block_sptr const& config );
  virtual ~detect_in_subregions_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class detect_in_subregions_process



} // end namespace

#endif /* _VIAME_DETECT_IN_SUBREGIONS_PROCESS_H */
