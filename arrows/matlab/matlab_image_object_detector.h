// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header defining matlab image object detector
 */

#ifndef VITAL_BINDINGS_MATLAB_IMAGE_OBJECT_DETECTOR_H
#define VITAL_BINDINGS_MATLAB_IMAGE_OBJECT_DETECTOR_H

#include <vital/algo/image_object_detector.h>
#include <arrows/matlab/kwiver_algo_matlab_export.h>

namespace kwiver {
namespace arrows {
namespace matlab {

class KWIVER_ALGO_MATLAB_EXPORT matlab_image_object_detector
  : public vital::algo::image_object_detector
{
public:
  matlab_image_object_detector();
  virtual ~matlab_image_object_detector();

  PLUGIN_INFO( "matlab",
               "Wrapper/bridge to matlab object detector implementation." );

  vital::config_block_sptr get_configuration() const override;
  void set_configuration(vital::config_block_sptr config) override;
  bool check_configuration(vital::config_block_sptr config) const override;

  // Main detection method
  vital::detected_object_set_sptr detect( vital::image_container_sptr image_data) const override;

private:
  class priv;
  const std::unique_ptr<priv> d;
};

} } } // end namespace

#endif // VITAL_BINDINGS_MATLAB_IMAGE_OBJECT_DETECTOR_H
