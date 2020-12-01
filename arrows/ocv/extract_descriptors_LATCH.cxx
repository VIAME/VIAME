// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV LATCH descriptor extractor wrapper implementation
 */

#include "extract_descriptors_LATCH.h"

#ifdef HAVE_OPENCV_XFEATURES2D

#include <opencv2/xfeatures2d.hpp>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {

class extract_descriptors_LATCH::priv
{
public:
  priv()
    : bytes( 32 )
    , rotation_invariance( true )
    , half_ssd_size( 3 )
  {
  }

  cv::Ptr<cv::xfeatures2d::LATCH> create() const
  {
    return cv::xfeatures2d::LATCH::create( bytes, rotation_invariance,
                                           half_ssd_size );
  }

  void update_config( config_block_sptr config ) const
  {
    config->set_value( "bytes", bytes, "" );
    config->set_value( "rotation_invariance", rotation_invariance, "" );
    config->set_value( "half_ssd_size", half_ssd_size, "" );
  }

  void set_config( config_block_sptr config )
  {
    bytes = config->get_value<int>( "bytes" );
    rotation_invariance = config->get_value<bool>( "rotation_invariance" );
    half_ssd_size = config->get_value<int>( "half_ssd_size" );
  }

  bool check_config( config_block_sptr config, logger_handle_t const &log ) const
  {
    bool valid = true;

    // Bytes can only be one of the following values
    int bytes_ = config->get_value<int>( "bytes" );
    if( ! ( bytes_ == 1 ||
            bytes_ == 2 ||
            bytes_ == 4 ||
            bytes_ == 8 ||
            bytes_ == 16 ||
            bytes_ == 32 ||
            bytes_ == 64 ) )
    {
      LOG_ERROR( log, "bytes value must be one of [1, 2, 4, 8, 16, 32, 64]. "
                      "Given: " << bytes_ );
      valid = false;
    }

    return valid;
  }

  // Parameters
  int bytes;
  bool rotation_invariance;
  int half_ssd_size;
};

extract_descriptors_LATCH
::extract_descriptors_LATCH()
  : p_( new priv )
{
  attach_logger( "arrows.ocv.LATCH" );
  extractor = p_->create();
}

extract_descriptors_LATCH
::~extract_descriptors_LATCH()
{
}

vital::config_block_sptr
extract_descriptors_LATCH
::get_configuration() const
{
  config_block_sptr config = ocv::extract_descriptors::get_configuration();
  p_->update_config( config );
  return config;
}

void extract_descriptors_LATCH
::set_configuration(vital::config_block_sptr config)
{
  config_block_sptr c = get_configuration();
  c->merge_config( config );
  p_->set_config( c );
  extractor = p_->create();
}

bool
extract_descriptors_LATCH
::check_configuration(vital::config_block_sptr config) const
{
  config_block_sptr c = get_configuration();
  c->merge_config( config );
  return p_->check_config( c, logger() );
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif //HAVE_OPENCV_XFEATURES2D
