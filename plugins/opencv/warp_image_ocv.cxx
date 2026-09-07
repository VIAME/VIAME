/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation of OCV warp image algorithm
 */

#include "warp_image_ocv.h"

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace viame {

namespace kv = kwiver::vital;
namespace ocv = kwiver::arrows::ocv;

namespace {

double
depth_max_value( int depth )
{
  switch( depth )
  {
    case CV_8U:  return 255.0;
    case CV_8S:  return 127.0;
    case CV_16U: return 65535.0;
    case CV_16S: return 32767.0;
    default:     return 1.0;
  }
}

} // end anonymous namespace

/// Warp image
kv::image_container_sptr
warp_image_ocv
::warp( kv::image_container_sptr src_image,
        kv::image_container_sptr dst_image,
        kv::homography_sptr homography,
        kv::image_container_sptr alpha_mask ) const
{
  if( !src_image || !homography )
  {
    return dst_image;
  }

  cv::Mat source =
    ocv::image_container::vital_to_ocv(
      src_image->get_image(), ocv::image_container::BGR_COLOR );

  cv::Mat dest = dst_image ?
    ocv::image_container::vital_to_ocv(
      dst_image->get_image(), ocv::image_container::BGR_COLOR ).clone() :
    cv::Mat::zeros( source.size(), source.type() );

  Eigen::Matrix< double, 3, 3 > const eigen_matrix = homography->matrix();
  cv::Mat matrix;
  cv::eigen2cv( eigen_matrix, matrix );

  cv::Mat warped;
  cv::warpPerspective( source, warped, matrix, dest.size() );

  if( warped.depth() != dest.depth() )
  {
    warped.convertTo( warped, dest.depth() );
  }

  if( alpha_mask )
  {
    cv::Mat mask =
      ocv::image_container::vital_to_ocv(
        alpha_mask->get_image(), ocv::image_container::BGR_COLOR );

    cv::Mat weight;
    mask.convertTo( weight, CV_32F, 1.0 / depth_max_value( mask.depth() ) );

    cv::Mat warped_weight;
    cv::warpPerspective( weight, warped_weight, matrix, dest.size() );

    cv::Mat weights;
    cv::merge( std::vector< cv::Mat >( dest.channels(), warped_weight ), weights );

    cv::Mat warped_float, dest_float;
    warped.convertTo( warped_float, CV_32F );
    dest.convertTo( dest_float, CV_32F );

    cv::Mat blended =
      warped_float.mul( weights ) + dest_float.mul( 1.0 - weights );

    blended.convertTo( dest, dest.type() );
  }
  else
  {
    cv::Mat covered;
    cv::warpPerspective(
      cv::Mat( source.size(), CV_8UC1, cv::Scalar( 255 ) ),
      covered, matrix, dest.size(), cv::INTER_NEAREST );

    warped.copyTo( dest, covered );
  }

  return std::make_shared< ocv::image_container >(
    dest, ocv::image_container::BGR_COLOR );
}

} // end namespace viame
