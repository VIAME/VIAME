// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV BRIEF descriptor extractor wrapper implementation
 */

#include "extract_descriptors_BRIEF.h"

#if KWIVER_OPENCV_VERSION_MAJOR < 3 || defined(HAVE_OPENCV_XFEATURES2D)

#include <sstream>

#if KWIVER_OPENCV_VERSION_MAJOR < 3
typedef cv::BriefDescriptorExtractor cv_BRIEF_t;
#else
#include <opencv2/xfeatures2d.hpp>
typedef cv::xfeatures2d::BriefDescriptorExtractor cv_BRIEF_t;
#endif

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {

class extract_descriptors_BRIEF::priv
{
public:
  /// Constructor
  priv()
    : bytes( 32 )
#if KWIVER_OPENCV_VERSION_MAJOR >= 3
    , use_orientation( false )
#endif
  {
  }

  /// Create new algorithm instance using current parameter values
  cv::Ptr<cv_BRIEF_t> create() const
  {
#if KWIVER_OPENCV_VERSION_MAJOR < 3
    return cv::Ptr<cv_BRIEF_t>( new cv_BRIEF_t(bytes) );
#else
    return cv_BRIEF_t::create( bytes, use_orientation );
#endif
  }

#if KWIVER_OPENCV_VERSION_MAJOR < 3
  /// Update given algorithm using current parameter values
  void update(cv::Ptr<cv_BRIEF_t> descriptor) const
  {
    descriptor->set( "bytes", bytes );
  }
#endif

  void update_config( config_block_sptr config ) const
  {
    config->set_value( "bytes", bytes,
                       "Length of descriptor in bytes. It can be equal 16, 32 "
                       "or 64 bytes." );
#if KWIVER_OPENCV_VERSION_MAJOR >= 3
    config->set_value( "use_orientation", use_orientation,
                       "sample patterns using keypoints orientation, disabled "
                       "by default." );
#endif
  }

  void set_config( config_block_sptr config )
  {
    bytes = config->get_value<int>( "bytes" );
#if KWIVER_OPENCV_VERSION_MAJOR >= 3
    use_orientation = config->get_value<bool>( "use_orientation" );
#endif
  }

  bool check_config( config_block_sptr config, logger_handle_t const &logger ) const
  {
    bool valid = true;

    // check that bytes param is one of the required 3 values
    int b = config->get_value<int>( "bytes" );
    if( ! ( b == 16 || b == 32 || b == 64 ) )
    {
      LOG_ERROR( logger,
                 "Bytes parameter must be either 16, 32 or 64. Given: " << b );
      valid = false;
    }

    return valid;
  }

  // Parameters
  int bytes;
#if KWIVER_OPENCV_VERSION_MAJOR >= 3
  bool use_orientation;
#endif
};

/// Constructor
extract_descriptors_BRIEF
::extract_descriptors_BRIEF()
  : p_(new priv)
{
  attach_logger("arrows.ocv.BRIEF");
  extractor = p_->create();
}

/// Destructor
extract_descriptors_BRIEF
::~extract_descriptors_BRIEF()
{
}

vital::config_block_sptr
extract_descriptors_BRIEF
::get_configuration() const
{
  vital::config_block_sptr config =
      ocv::extract_descriptors::get_configuration();
  p_->update_config( config );
  return config;
}

void
extract_descriptors_BRIEF
::set_configuration(vital::config_block_sptr config)
{
  vital::config_block_sptr c = get_configuration();
  c->merge_config( config );
  p_->set_config( c );

#if KWIVER_OPENCV_VERSION_MAJOR < 3
  p_->update( extractor );
#else
  extractor = p_->create();
#endif
}

bool
extract_descriptors_BRIEF
::check_configuration(vital::config_block_sptr in_config) const
{
  vital::config_block_sptr config = get_configuration();
  config->merge_config(in_config);
  return p_->check_config( config, logger() );
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif // has OCV support
