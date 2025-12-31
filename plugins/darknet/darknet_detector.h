/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_DARKNET_DETECTOR_H
#define VIAME_DARKNET_DETECTOR_H

#include "viame_darknet_export.h"

#include <vital/algo/image_object_detector.h>

namespace viame {

class VIAME_DARKNET_EXPORT darknet_detector :
  public kwiver::vital::algo::image_object_detector
{
public:
  darknet_detector();
  virtual ~darknet_detector();

  PLUGIN_INFO( "darknet",
               "Image object detector using darknet." )

  virtual kwiver::vital::config_block_sptr get_configuration() const;

  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual kwiver::vital::detected_object_set_sptr detect(
    kwiver::vital::image_container_sptr image_data ) const;

private:
  class priv;
  const std::unique_ptr< priv > d;
};

} // end namespace

#endif /* VIAME_DARKNET_DETECTOR_H */
