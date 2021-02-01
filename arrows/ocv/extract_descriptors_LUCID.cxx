// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV LUCID descriptor extractor wrapper implementation
 */

#include "extract_descriptors_LUCID.h"

#include <vital/vital_config.h>

#ifdef HAVE_OPENCV_XFEATURES2D

#include <opencv2/xfeatures2d.hpp>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {

class extract_descriptors_LUCID::priv
{
public:
  priv()
    : lucid_kernel( 1 )
    , blur_kernel( 1 )
  {
  }

  cv::Ptr<cv::xfeatures2d::LUCID> create() const
  {
    return cv::xfeatures2d::LUCID::create( lucid_kernel, blur_kernel );
  }

  void update_config( config_block_sptr config ) const
  {
    config->set_value( "lucid_kernel", lucid_kernel,
                       "kernel for descriptor construction, where 1=3x3, "
                       "2=5x5, 3=7x7 and so forth" );
    config->set_value( "blur_kernel", blur_kernel,
                       "kernel for blurring image prior to descriptor "
                       "construction, where 1=3x3, 2=5x5, 3=7x7 and so forth" );
  }

  void set_config( config_block_sptr config )
  {
    lucid_kernel = config->get_value<int>( "lucid_kernel" );
    blur_kernel = config->get_value<int>( "blur_kernel" );
  }

  // Parameters
  int lucid_kernel;
  int blur_kernel;
};

extract_descriptors_LUCID
::extract_descriptors_LUCID()
  : p_( new priv )
{
  attach_logger( "arrows.ocv.LUCID" );
  extractor = p_->create();
}

extract_descriptors_LUCID
::~extract_descriptors_LUCID()
{
}

vital::config_block_sptr
extract_descriptors_LUCID
::get_configuration() const
{
  config_block_sptr config = ocv::extract_descriptors::get_configuration();
  p_->update_config( config );
  return config;
}

void extract_descriptors_LUCID
::set_configuration(vital::config_block_sptr config)
{
  config_block_sptr c = get_configuration();
  c->merge_config( config );
  p_->set_config( c );
  extractor = p_->create();
}

bool
extract_descriptors_LUCID
::check_configuration( VITAL_UNUSED vital::config_block_sptr config ) const
{
  return true;
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif //HAVE_OPENCV_XFEATURES2D
