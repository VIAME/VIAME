/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_CONVERT_HEAD_TAIL_POINTS_H
#define VIAME_CORE_CONVERT_HEAD_TAIL_POINTS_H

#include "viame_core_export.h"

#include <vital/algo/refine_detections.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

class VIAME_CORE_EXPORT convert_head_tail_points :
  public kwiver::vital::algo::refine_detections
{
public:
  PLUGGABLE_IMPL(
    convert_head_tail_points,
    "This process converts between different methods for storing head and tail "
    "points within object detections, most commonly from seperate detections to "
    "attributes within detections.",
    PARAM_DEFAULT(
      head_postfix, std::string,
      "Detection type postfix indicating head position.",
      "_head" ),
    PARAM_DEFAULT(
      tail_postfix, std::string,
      "Detection type postfix indicating tail position.",
      "_tail" )
  )

  virtual ~convert_head_tail_points() = default;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  // Main detection method
  virtual kwiver::vital::detected_object_set_sptr refine(
    kwiver::vital::image_container_sptr image_data,
    kwiver::vital::detected_object_set_sptr input_dets ) const;
};

} // end namespace

#endif /* VIAME_CONVERT_HEAD_TAIL_POINTS_H */
