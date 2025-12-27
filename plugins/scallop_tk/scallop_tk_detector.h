/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_SCALLOP_TK_DETECTOR_H
#define VIAME_SCALLOP_TK_DETECTOR_H

#include <plugins/scallop_tk/viame_scallop_tk_export.h>

#include <vital/algo/image_object_detector.h>

namespace viame {

class VIAME_SCALLOP_TK_EXPORT scallop_tk_detector :
  public kwiver::vital::algorithm_impl<
    scallop_tk_detector, kwiver::vital::algo::image_object_detector >
{
public:
  scallop_tk_detector();
  virtual ~scallop_tk_detector();

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

#endif /* VIAME_SCALLOP_TK_DETECTOR_H */
