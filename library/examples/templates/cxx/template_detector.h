/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_@TEMPLATE@_DETECTOR_H
#define VIAME_@TEMPLATE@_DETECTOR_H

#include "viame_@template_lib@_export.h"

#include <viame/algorithm_framework/algo/image_object_detector.h>
#include <viame/algorithm_framework/plugin/pluggable_macro_magic.h>

namespace viame {

// -----------------------------------------------------------------------------
/**
 * @brief @template@ detector
 *
 * `PLUGGABLE_IMPL` declares the name the detector registers under, its
 * description and its configuration: each `PARAM_DEFAULT` becomes a config
 * key with that default, a `c_<name>` member and a `get_<name>()` accessor,
 * and `get_configuration` / `set_configuration` are generated from the list.
 */
class VIAME_@TEMPLATE_LIB@_EXPORT @template@_detector
  : public viame::algo::image_object_detector
{
public:
  PLUGGABLE_IMPL(
    @template@_detector,
    "@template@ detector",
    PARAM_DEFAULT(
      text, std::string,
      "Text to display to user.",
      "Hello World" )
  )

  virtual ~@template@_detector() = default;

  // Check for anything that would stop the detector from running
  virtual bool check_configuration(
    viame::config_block_sptr config ) const override;

  // Main detection method
  virtual viame::detected_object_set_sptr detect(
    viame::image_container_sptr image_data ) const override;
};

} // end namespace viame

#endif // VIAME_@TEMPLATE@_DETECTOR_H
