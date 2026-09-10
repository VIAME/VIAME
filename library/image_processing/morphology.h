/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_PROCESSING_MORPHOLOGY_H
#define VIAME_IMAGE_PROCESSING_MORPHOLOGY_H

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/image_filter.h>
#include <viame/algorithm_framework/plugin/pluggable_macro_magic.h>

namespace viame {

namespace kv = kwiver::vital;

/// @brief Erode, dilate, open or close a binary mask.
///
/// Config keys, defaults and results match the `vxl_morphology` it replaces,
/// including the structuring element sizes: the disk comparison is strict, so
/// a radius of 1 is a single pixel and 2 is a 3x3 square.
class VIAME_IMAGE_PROCESSING_EXPORT morphology
  : public kv::algo::image_filter
{
public:
#define VIAME_MORPHOLOGY_PARAMS \
    PARAM_DEFAULT( \
      morphology, std::string, \
      "Morphological operation to apply: erode, dilate, open, close, none", \
      "dilate" ), \
    PARAM_DEFAULT( \
      element_shape, std::string, \
      "Shape of the structuring element: disk, iline, jline", \
      "disk" ), \
    PARAM_DEFAULT( \
      kernel_radius, double, \
      "Radius of the structuring element", \
      1.5 ), \
    PARAM_DEFAULT( \
      channel_combination, std::string, \
      "How to combine the channels afterwards: none, union, intersection", \
      "none" )

  // PLUGGABLE_IMPL_NAMED rather than the pieces spelled out: it is
  // the only spelling that also generates get_configuration, without
  // which a partial config from a pipe file throws on the first key
  // the file does not set
  PLUGGABLE_IMPL_NAMED(
    morphology,
    "morphology",
    "Apply a binary morphological operation to a mask",
    VIAME_MORPHOLOGY_PARAMS )

  virtual ~morphology();

  bool check_configuration( kv::config_block_sptr config ) const override;

  kv::image_container_sptr filter( kv::image_container_sptr image_data ) override;

  void set_configuration_internal( kv::config_block_sptr config ) override;

private:
  void initialize() override;

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace viame

#endif // VIAME_IMAGE_PROCESSING_MORPHOLOGY_H
