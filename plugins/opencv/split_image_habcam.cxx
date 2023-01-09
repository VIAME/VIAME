// This file is part of VIAME, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of habcam split image horizontally algorithm
 */

#include "split_image_habcam.h"

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace viame {

class split_image_habcam::priv
{
public:

  priv()
    : m_require_stereo( false )
    , m_required_width_factor( 2.0 )
  {}

  ~priv() {}

  bool m_require_stereo;
  double m_required_width_factor;
};


/// Constructor
split_image_habcam
::split_image_habcam()
  : d( new priv )
{
}

/// Destructor
split_image_habcam
::~split_image_habcam()
{
}

/// Configs
kwiver::vital::config_block_sptr
split_image_habcam
::get_configuration() const
{
  kwiver::vital::config_block_sptr config =
    kwiver::vital::algorithm::get_configuration();

  config->set_value( "require_stereo",
    d->m_require_stereo,
    "Fail if the input is not a conjoined stereo image pair" );
  config->set_value( "required_width_factor",
    d->m_required_width_factor,
    "If the width is this time as many heights, it is a stereo pair." );

  return config;
}


void
split_image_habcam
::set_configuration( kwiver::vital::config_block_sptr config_in )
{
  kwiver::vital::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  d->m_require_stereo =
    config->get_value< bool >( "require_stereo" );
  d->m_required_width_factor =
    config->get_value< double >( "required_width_factor" );
}


bool
split_image_habcam
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


/// Split image
std::vector< kwiver::vital::image_container_sptr >
split_image_habcam
::split( kwiver::vital::image_container_sptr image ) const
{
  std::vector< kwiver::vital::image_container_sptr > output;

  if( image->width() >= d->m_required_width_factor * image->height() )
  {
    cv::Mat cv_image =
      kwiver::arrows::ocv::image_container::vital_to_ocv(
        image->get_image(),
        kwiver::arrows::ocv::image_container::RGB_COLOR );

    cv::Mat left_image =
      cv_image(
        cv::Rect( 0, 0, cv_image.cols / 2, cv_image.rows ) );
    cv::Mat right_image =
      cv_image(
        cv::Rect( cv_image.cols / 2, 0, cv_image.cols / 2, cv_image.rows ) );

    output.push_back(
      kwiver::vital::image_container_sptr(
        new kwiver::arrows::ocv::image_container( left_image.clone(),
        kwiver::arrows::ocv::image_container::RGB_COLOR ) ) );
    output.push_back(
      kwiver::vital::image_container_sptr(
        new kwiver::arrows::ocv::image_container( right_image.clone(),
        kwiver::arrows::ocv::image_container::RGB_COLOR ) ) );
  }
  else
  {
    output.push_back( image );
  }

  return output;
}

} // end namespace viame
