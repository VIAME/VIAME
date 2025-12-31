/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_@TEMPLATE@_DETECTOR_H
#define VIAME_@TEMPLATE@_DETECTOR_H

#include "viame_@template_lib@_export.h"

#include <vital/algo/image_object_detector.h>

namespace viame {

class VIAME_@TEMPLATE_LIB@_EXPORT @template@_detector :
  public kwiver::vital::algorithm_impl<
    @template@_detector, kwiver::vital::algo::image_object_detector >
{
public:
  @template@_detector();
  virtual ~@template@_detector();

  // Get the current configuration (parameters) for this detector
  virtual kwiver::vital::config_block_sptr get_configuration() const;

  // Set configurations automatically parsed from input pipeline and config files
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  // Main detection method
  virtual kwiver::vital::detected_object_set_sptr detect(
    kwiver::vital::image_container_sptr image_data ) const;

private:
  class priv;
  const std::unique_ptr< priv > d;
};

} // end namespace

#endif // VIAME_@TEMPLATE@_DETECTOR_H
