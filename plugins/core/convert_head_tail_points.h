 /* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_CONVERT_HEAD_TAIL_POINTS_H
#define VIAME_CORE_CONVERT_HEAD_TAIL_POINTS_H

#include "viame_core_export.h"

#include <vital/algo/refine_detections.h>

namespace viame {

class VIAME_CORE_EXPORT convert_head_tail_points :
  public kwiver::vital::algo::refine_detections
{
public:
  convert_head_tail_points();
  virtual ~convert_head_tail_points();

  static constexpr char const* name = "convert_head_tail_points";

  static constexpr char const* description =
    "This process converts between different methods for storing head and tail "
    "points within object detections, most commonly from seperate detections to "
    "attributes within detections.";

  // Get the current configuration (parameters) for this detector
  virtual kwiver::vital::config_block_sptr get_configuration() const;

  // Set configurations automatically parsed from input pipeline and config files
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  // Main detection method
  virtual kwiver::vital::detected_object_set_sptr refine(
    kwiver::vital::image_container_sptr image_data,
    kwiver::vital::detected_object_set_sptr input_dets ) const;

private:
  class priv;
  const std::unique_ptr< priv > d;
};

} // end namespace

#endif /* VIAME_CONVERT_HEAD_TAIL_POINTS_H */
