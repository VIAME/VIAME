// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV BRISK feature detector and extractor wrapper implementation
 */

#include "feature_detect_extract_BRISK.h"

#include <vital/vital_config.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {

namespace {

/// Common BRISK private implementation class
/**
 * TODO: Support for custom pattern configuration and constructor?
 */
class priv
{
public:
  /// Constructor
  priv()
     : threshold(30),
       octaves(3),
       pattern_scale(1.0f)
  {
  }

  /// Create new impl instance based on current parameters
  cv::Ptr<cv::BRISK> create() const
  {
#if KWIVER_OPENCV_VERSION_MAJOR < 3
    return cv::Ptr<cv::BRISK>(
      new cv::BRISK( threshold, octaves, pattern_scale )
    );
#else
    return cv::BRISK::create( threshold, octaves, pattern_scale );
#endif
  }

  /// Update given configuration with current parameter keys and values
  void update_configuration(vital::config_block_sptr &config) const
  {
    config->set_value("threshold", threshold,
                      "AGAST detection threshold score.");
    config->set_value("octaves", octaves,
                      "detection octaves. Use 0 to do single scale.");
    config->set_value("pattern_scale", pattern_scale,
                      "apply this scale to the pattern used for sampling the "
                         "neighbourhood of a keypoint.");
  }

  /// Update parameters based on the given config-block
  void set_configuration(vital::config_block_sptr const &config)
  {
    threshold = config->get_value<int>("threshold");
    octaves = config->get_value<int>("octaves");
    pattern_scale = config->get_value<float>("pattern_scale");
  }

  /// Parameters
  int threshold;
  int octaves;
  float pattern_scale;
};

} // end anon namespace

/// Private implementation class for BRISK feature detection
class detect_features_BRISK::priv
  : public ocv::priv
{
};

/// Private implementation class for BRISK descriptor extraction
class extract_descriptors_BRISK::priv
  : public ocv::priv
{
};

detect_features_BRISK
::detect_features_BRISK()
  : p_( new priv )
{
  attach_logger("arrows.ocv.BRISK");
  detector = p_->create();
}

detect_features_BRISK
::~detect_features_BRISK()
{
}

vital::config_block_sptr
detect_features_BRISK
::get_configuration() const
{
  config_block_sptr config = detect_features::get_configuration();
  p_->update_configuration(config);
  return config;
}

void
detect_features_BRISK
::set_configuration(vital::config_block_sptr config)
{
  config_block_sptr c = get_configuration();
  c->merge_config(config);
  p_->set_configuration(c);
  detector = p_->create();
}

bool
detect_features_BRISK
::check_configuration( VITAL_UNUSED vital::config_block_sptr config ) const
{
  return true;
}

extract_descriptors_BRISK
::extract_descriptors_BRISK()
  : p_( new priv )
{
  attach_logger("arrows.ocv.BRISK");
  extractor = p_->create();
}

extract_descriptors_BRISK
::~extract_descriptors_BRISK()
{
}

vital::config_block_sptr
extract_descriptors_BRISK
::get_configuration() const
{
  config_block_sptr config = extract_descriptors::get_configuration();
  p_->update_configuration(config);
  return config;
}

void
extract_descriptors_BRISK
::set_configuration(vital::config_block_sptr config)
{
  config_block_sptr c = get_configuration();
  c->merge_config(config);
  p_->set_configuration(c);
  extractor = p_->create();
}

bool
extract_descriptors_BRISK
::check_configuration( VITAL_UNUSED vital::config_block_sptr config) const
{
  return true;
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
