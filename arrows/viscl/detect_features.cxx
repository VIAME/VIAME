// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "detect_features.h"

#include <vector>

#include <arrows/viscl/feature_set.h>
#include <arrows/viscl/image_container.h>

#include <viscl/tasks/hessian.h>

namespace kwiver {
namespace arrows {
namespace vcl {

/// Private implementation class
class detect_features::priv
{
public:
  /// Constructor
  priv() : max_kpts(5000), thresh(0.003f), sigma(2.0f)
  {
  }

  viscl::hessian detector;
  unsigned int max_kpts;
  float thresh;
  float sigma;
};

/// Constructor
detect_features
::detect_features()
: d_(new priv)
{
}

/// Destructor
detect_features
::~detect_features()
{
}

/// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
detect_features
::get_configuration() const
{
  vital::config_block_sptr config = algorithm::get_configuration();
  config->set_value("max_keypoints", d_->max_kpts, "Maximum number of features to detect on an image.");
  config->set_value("thresh", d_->thresh, "Threshold on the determinant of Hessian for keypoint candidates.");
  config->set_value("sigma", d_->sigma, "Smoothing scale.");
  return config;
}

/// Set this algorithm's properties via a config block
void
detect_features
::set_configuration(vital::config_block_sptr config)
{
  d_->max_kpts = config->get_value<unsigned int>("max_keypoints", d_->max_kpts);
  d_->thresh = config->get_value<float>("thresh", d_->thresh);
  d_->sigma = config->get_value<float>("sigma", d_->sigma);
}

/// Check that the algorithm's configuration vital::config_block is valid
bool
detect_features
::check_configuration(vital::config_block_sptr config) const
{
  return true;
}

/// Extract a set of image features from the provided image
/// \param image_data contains the image data to process
/// \returns a set of image features
vital::feature_set_sptr
detect_features
::detect(vital::image_container_sptr image_data, vital::image_container_sptr mask) const
{
  // TODO: Do something with the given mask

  viscl::image img = vcl::image_container_to_viscl(*image_data);
  vcl::feature_set::type feature_data;

  d_->detector.smooth_and_detect(img, feature_data.kptmap_, feature_data.features_, feature_data.numfeat_,
                                 d_->max_kpts, d_->thresh, d_->sigma);

  return vital::feature_set_sptr(new feature_set(feature_data));
}

} // end namespace vcl
} // end namespace arrows
} // end namespace kwiver
