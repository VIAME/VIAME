// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV brute-force feature matcher wrapper implementation
 */

#include "match_features_bruteforce.h"

namespace kwiver {
namespace arrows {
namespace ocv {

class match_features_bruteforce::priv
{
public:
  priv(int p_norm_type = cv::NORM_L2, bool p_cross_check = false)
      : norm_type( p_norm_type ),
        cross_check( p_cross_check ),
        matcher( new cv::BFMatcher( norm_type, cross_check ) )
  {
  }

  // Can't currently update parameters on BF implementation, so no update
  // function. Will need to create a new instance on each parameter update.

  /// Create a new brute-force matcher instance and set our matcher param to it
  void create()
  {
    // cross version compatible
    matcher = cv::Ptr<cv::BFMatcher>(
        new cv::BFMatcher(norm_type, cross_check)
    );
  }

  /// Parameters
  int norm_type;
  bool cross_check;
  cv::Ptr<cv::BFMatcher> matcher;

}; // end match_features_bruteforce::priv

namespace {

/// Norm type info string generator
std::string str_list_enum_values()
{
  std::stringstream ss;
  ss << "cv::NORM_INF="       << cv::NORM_INF       << ", "
     << "cv::NORM_L1="        << cv::NORM_L1        << ", "
     << "cv::NORM_L2="        << cv::NORM_L2        << ", "
     << "cv::NORM_L2SQR="     << cv::NORM_L2SQR     << ", "
     << "cv::NORM_HAMMING="   << cv::NORM_HAMMING   << ", "
     << "cv::NORM_HAMMING2="  << cv::NORM_HAMMING2  << ", "
     << "cv::NORM_TYPE_MASK=" << cv::NORM_TYPE_MASK << ", "
     << "cv::NORM_RELATIVE="  << cv::NORM_RELATIVE  << ", "
     << "cv::NORM_MINMAX="    << cv::NORM_MINMAX;
  return ss.str();
}

/// Check value against known OCV norm enum values
bool check_norm_enum_value(int norm_type)
{
  switch( norm_type )
  {
    case cv::NORM_INF:
    case cv::NORM_L1:
    case cv::NORM_L2:
    case cv::NORM_L2SQR:
    case cv::NORM_HAMMING:
    case cv::NORM_HAMMING2:
    //case cv::NORM_TYPE_MASK:  // This is the same value as HAMMING2 apparently
    case cv::NORM_RELATIVE:
    case cv::NORM_MINMAX:
      return true;
    default:
      return false;
  }
}

}

match_features_bruteforce
::match_features_bruteforce()
  : p_( new priv )
{
  attach_logger( "arrows.ocv.match_features_bruteforce" );
}

match_features_bruteforce
::~match_features_bruteforce()
{
}

vital::config_block_sptr
match_features_bruteforce
::get_configuration() const
{
  vital::config_block_sptr config = match_features::get_configuration();

  config->set_value( "cross_check", p_->cross_check,
                     "Perform cross checking when finding matches to filter "
                     "through only the consistent pairs. This is an "
                     "alternative to the ratio test used by D. Lowe in the "
                     "SIFT paper." );

  std::stringstream ss;
  ss << "Normalization type enum value. This should be one of the enum values: "
     << str_list_enum_values();
  config->set_value( "norm_type", p_->norm_type, ss.str());

  return config;
}

void
match_features_bruteforce
::set_configuration(vital::config_block_sptr in_config)
{
  vital::config_block_sptr config = get_configuration();
  config->merge_config(in_config);

  p_->cross_check = config->get_value<bool>("cross_check");
  p_->norm_type = config->get_value<int>("norm_type");

  // Create new instance with the updated parameters
  p_->create();
}

bool
match_features_bruteforce
::check_configuration(vital::config_block_sptr in_config) const
{
  vital::config_block_sptr config = get_configuration();
  config->merge_config(in_config);
  bool valid = true;

  // user has the chance to input an incorret value for the norm type enum value
  int norm_type = config->get_value<int>( "norm_type" );
  if( ! check_norm_enum_value( norm_type ) )
  {
    std::stringstream ss;
    ss << "Incorrect norm type enum value given: '" << norm_type << "'. "
       << "Valid values are: " << str_list_enum_values();
    logger()->log_error( ss.str() );
    valid = false;
  }

  return valid;
}

void
match_features_bruteforce
::ocv_match(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
            std::vector<cv::DMatch> &matches) const
{
  p_->matcher->match( descriptors1, descriptors2, matches );
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
