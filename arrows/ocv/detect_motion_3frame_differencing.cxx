// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of ocv::detect_moiion_3frame_differencing
 */

#include <deque>

#include "detect_motion_3frame_differencing.h"

#include <kwiversys/SystemTools.hxx>
#include <vital/exceptions.h>
#include <vital/types/matrix.h>
#include <vital/vital_config.h>

#include <arrows/ocv/image_container.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

namespace kwiver {
namespace arrows {
namespace ocv {

using namespace kwiver::vital;

//-----------------------------------------------------------------------------
  ///
  /**
  * @brief Converts a multi-channel image into a single channel image
  *
  * Replace multi-channel image with a single channel image equal to the root
  * mean square over the channels.
  *
  * @param src first image
  * @param dst second image
  */
static
void
rms_over_channels( const cv::Mat &src, cv::Mat &dst)
{
  cv::Mat *src_split = new cv::Mat[src.channels()];
  cv::split(src, src_split);
  cv::Mat accum = cv::Mat(src.rows, src.cols, CV_32F, cv::Scalar(0));
  for( int i=0; i<src.channels(); ++i)
  {
    cv::Mat temp;
    src_split[i].convertTo(temp, CV_32F);

    // Divide the result by 3^2 so that the difference has the same scale as
    // a mono image
    cv::multiply( temp, temp, temp, 1/3.0 );
    accum += temp;
  }
  cv::sqrt(accum, accum);
  accum.convertTo(dst, CV_8UC1);
  delete [] src_split;
}

// ----------------------------------------------------------------------------
/// Private implementation class
class detect_motion_3frame_differencing::priv
{
  cv::Mat m_jitter_struct_el;
  std::deque<cv::Mat> m_frames;
  int m_debug_counter = 0;

public:
  /// Parameters
  std::string m_debug_dir;
  bool m_output_to_debug_dir = false;
  std::size_t m_frame_separation;
  int m_jitter_radius;
  double m_max_foreground_fract;
  double m_max_foreground_fract_thresh;
  kwiver::vital::logger_handle_t m_logger;

  /// Constructor
  priv()
     :
       m_frame_separation(1),
       m_jitter_radius(0),
       m_max_foreground_fract(1),
       m_max_foreground_fract_thresh(-1)
  {
  }

  /// Flush the image queue.
  void reset()
  {
    m_frames.clear();
  }

  ///
  /**
  * @brief Calculates a jittered difference img1 and img2
  *
  * For each pixel in img1, the minimum absolute difference ||img1-b|| is
  * calculated, where b is drawn from a neighborhood (defined by
  * m_jitter_radius) around the equivalent pixel in img2.
  *
  * @param img1 first image
  * @param img2 second image
  * @param img_diff difference image
  */
  void
  image_difference( const cv::Mat &img1, const cv::Mat &img2, cv::Mat &img_diff )
  {
    if( m_jitter_radius == 0 )
    {
      cv::absdiff( img1, img2, img_diff );
    }
    else
    {
      if( m_jitter_struct_el.empty() )
      {
        cv::Size el_size(2*m_jitter_radius+1,2*m_jitter_radius+1);
        m_jitter_struct_el = cv::getStructuringElement( cv::MORPH_RECT, el_size);
      }
      cv::Mat local_max, local_min;
      cv::dilate( img2, local_max, m_jitter_struct_el );
      cv::erode( img2, local_min, m_jitter_struct_el );

      // Following the reference "Detecting and Tracking All Moving Objects in
      // Wide-Area Aerial Video" equation 2
      cv::Mat img2_min_minus_img1, img1_minus_img2_max;
      cv::subtract( local_min, img1, img2_min_minus_img1, cv::noArray(), CV_32F );
      cv::subtract( img1, local_max, img1_minus_img2_max, cv::noArray(), CV_32F );

      // Set negative values to zero
      cv::threshold( img2_min_minus_img1, img2_min_minus_img1, 0, 1, cv::THRESH_TOZERO );
      cv::threshold( img1_minus_img2_max, img1_minus_img2_max, 0, 1, cv::THRESH_TOZERO );

      img_diff = cv::max( img2_min_minus_img1, img1_minus_img2_max );
    }
  }

  void
  process_image(cv::Mat &cv_src, cv::Mat &fgmask)
  {
    // Images are in temporal order A (oldest), B, C (newest).
    cv::Mat imgA, imgB, imgC;
    cv_src.copyTo(imgC);
    m_frames.push_front(imgC);

    if( m_frames.size() < 2*m_frame_separation )
    {
      LOG_TRACE( m_logger, "Haven't collected enough frames yet, so setting "
                           "foreground mask to all zeros.");
      fgmask = cv::Mat(cv_src.rows, cv_src.cols, CV_8UC1, cv::Scalar(0));
      return;
    }

    LOG_TRACE( m_logger, "Getting frame from end of queue");
    imgA = m_frames.back();
    LOG_TRACE( m_logger, "Getting frame at index frame_separation");
    imgB = m_frames[m_frame_separation];

    if( m_frames.size() > 2*m_frame_separation )
    {
      LOG_TRACE( m_logger, "Removing frame from end of queue");
      m_frames.pop_back();
    }

    // unsigned_sum (default)
    ///  = | | A - C | + | C - B | - | A - B | |
    cv::Mat AminusC, CminusB, AminusB;
    image_difference( imgA, imgC, AminusC );
    image_difference( imgC, imgB, CminusB );
    image_difference( imgA, imgB, AminusB );

    fgmask = cv::abs( AminusC + CminusB - AminusB );

    if( m_output_to_debug_dir )
    {
      std::string fname;
      cv::Mat img;
      imgA.convertTo(img, CV_8UC1);
      fname = m_debug_dir + "/" + std::to_string(m_debug_counter) + "imgA" + ".tif";
      cv::imwrite( fname, img );
      imgB.convertTo(img, CV_8UC1);
      fname = m_debug_dir + "/" + std::to_string(m_debug_counter) + "imgB" + ".tif";
      cv::imwrite( fname, img );
      imgC.convertTo(img, CV_8UC1);
      fname = m_debug_dir + "/" + std::to_string(m_debug_counter) + "imgC" + ".tif";
      cv::imwrite( fname, img );
      AminusC.convertTo(img, CV_8UC1);
      fname = m_debug_dir + "/" + std::to_string(m_debug_counter) + "AminusC" + ".tif";
      cv::imwrite( fname, img );
      CminusB.convertTo(img, CV_8UC1);
      fname = m_debug_dir + "/" + std::to_string(m_debug_counter) + "CminusB" + ".tif";
      cv::imwrite( fname, img );
      AminusB.convertTo(img, CV_8UC1);
      fname = m_debug_dir + "/" + std::to_string(m_debug_counter) + "AminusB" + ".tif";
      cv::imwrite( fname, img );
      fgmask.convertTo(img, CV_8UC1);
      fname = m_debug_dir + "/" + std::to_string(m_debug_counter) + "fgmask" + ".tif";
      cv::imwrite( fname, img );
      ++m_debug_counter;
    }

    if( fgmask.channels() > 1 )
    {
      LOG_TRACE( m_logger, "Converting multichannel foreground mask to single "
                 "channel");
      rms_over_channels( fgmask, fgmask);
    }

    if ( IS_TRACE_ENABLED( m_logger ) )
    {
      double min_val, max_val;
      cv::minMaxLoc(fgmask, &min_val, &max_val);
      LOG_TRACE( m_logger, "heat map min: " + std::to_string( min_val ) +
                 " max: " + std::to_string( max_val ) );
    }

    if( m_max_foreground_fract < 1)
    {
      int total_pixels = fgmask.rows*fgmask.cols;
      int max_fg_pixels = total_pixels*m_max_foreground_fract;
      cv::Mat mask;
      cv::threshold( fgmask, mask, m_max_foreground_fract_thresh, 1,
                     cv::THRESH_BINARY );
      int nonzero_pixels = cv::sum(mask).val[0];
      LOG_TRACE( m_logger, (double)nonzero_pixels/(double)total_pixels*100 <<
                 "% foreground pixels." );
      if( nonzero_pixels > max_fg_pixels )
      {
        LOG_DEBUG( m_logger, "Foreground pixels exceed maximum set to " <<
                   m_max_foreground_fract*100 << "%, something must have "
                   "failed. Resetting background model." );

        // Reset background model, but wait until next iteration to start
        // updating it because the current frame might be bad.
        reset();
        fgmask = cv::Scalar(0);
      }
    }
  }

  /// Set up debug directory
  void
  setup_debug_dir()
  {
    LOG_DEBUG( m_logger, "Creating debug directory: " + m_debug_dir);
    kwiversys::SystemTools::MakeDirectory(m_debug_dir);
    m_output_to_debug_dir = true;
  }
};

/// Constructor
detect_motion_3frame_differencing
::detect_motion_3frame_differencing()
: d_(new priv)
{
  attach_logger( "arrows.ocv.detect_motion_3frame_differencing" );
  d_->m_logger = logger();
  d_->reset();
}

/// Destructor
detect_motion_3frame_differencing
::~detect_motion_3frame_differencing() noexcept
{
}

/// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
detect_motion_3frame_differencing
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "frame_separation", d_->m_frame_separation,
                     "Number of frames of separation for difference "
                     "calculation. Queue of collected images must be twice this "
                     "value before a three-frame difference can be "
                     "calculated." );
  config->set_value( "jitter_radius", d_->m_jitter_radius,
                     "Radius of jitter displacement (pixels) expected in the "
                     "image due to imperfect stabilization. The image "
                     "differencing process will search for the lowest-magnitude "
                     "difference in a neighborhood with radius equal to "
                     "jitter_radius." );
  config->set_value( "max_foreground_fract", d_->m_max_foreground_fract,
                     "Specifies the maximum expected fraction of the scene "
                     "that may contain foreground movers at any time. When the "
                     "fraction of pixels determined to be in motion exceeds "
                     "this value, the background model is assumed to be "
                     "invalid (e.g., due to excessive camera motion) and is "
                     "reset. The default value of 1 indicates that no checking "
                     "is done." );
  config->set_value( "max_foreground_fract_thresh",
                     d_->m_max_foreground_fract_thresh,
                     "To be used in conjunction with max_foreground_fract, this "
                     "parameter defines the threshold for foreground in order "
                     "to determine if the maximum fraction of foreground has "
                     "been exceeded." );
  config->set_value( "debug_dir", d_->m_debug_dir,
                     "Output debug images to this directory.");

  return config;
}

/// Set this algo's properties via a config block
void
detect_motion_3frame_differencing
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  d_->m_frame_separation   = config->get_value<int>( "frame_separation" );
  d_->m_jitter_radius   = config->get_value<int>( "jitter_radius" );
  d_->m_max_foreground_fract   = config->get_value<double>( "max_foreground_fract" );
  d_->m_max_foreground_fract_thresh   = config->get_value<double>( "max_foreground_fract_thresh" );
  d_->m_debug_dir         = config->get_value<std::string>( "debug_dir" );

  if( d_->m_frame_separation < 0 )
  {
    VITAL_THROW( algorithm_configuration_exception, type_name(), impl_name(),
                                             "frame_separation must be an "
                                             "integer greater than 0." );
  }

  if( d_->m_jitter_radius < 0 )
  {
    VITAL_THROW( algorithm_configuration_exception, type_name(), impl_name(),
                                             "m_jitter_radius must be an "
                                             "integer greater than 0." );
  }

  if( d_->m_max_foreground_fract < 0 || d_->m_max_foreground_fract > 1 )
  {
    VITAL_THROW( algorithm_configuration_exception, type_name(), impl_name(),
                                             "max_foreground_fract must be in "
                                             "the range 0-1." );
  }

  if( d_->m_max_foreground_fract != 1 && d_->m_max_foreground_fract_thresh < 0 )
  {
    VITAL_THROW( algorithm_configuration_exception, type_name(), impl_name(),
                                             "max_foreground_fract_thresh must "
                                             "be set as a positive value." );
  }

  if ( !(d_->m_debug_dir.empty() || d_->m_debug_dir == "" ) )
  {
    d_->setup_debug_dir();
  }

  LOG_DEBUG( logger(), "frame_separation: " << std::to_string(d_->m_frame_separation) );
  LOG_DEBUG( logger(), "jitter_radius: " << std::to_string(d_->m_jitter_radius) );
  LOG_DEBUG( logger(), "max_foreground_fract: " << std::to_string(d_->m_max_foreground_fract) );
  LOG_DEBUG( logger(), "max_foreground_fract_thresh: " << std::to_string(d_->m_max_foreground_fract_thresh) );
  LOG_DEBUG( logger(), "debug_dir: " << d_->m_debug_dir );
}

bool
detect_motion_3frame_differencing
::check_configuration( VITAL_UNUSED vital::config_block_sptr config ) const
{
  return true;
}

/// Detect motion from a sequence of images
image_container_sptr
detect_motion_3frame_differencing
::process_image( VITAL_UNUSED const timestamp& ts,
                 const image_container_sptr image,
                 bool reset_model)
{
  if ( !image)
  {
    VITAL_THROW( vital::invalid_data, "Inputs to ocv::detect_motion_3frame_differencing are null");
  }

  if( reset_model )
  {
    d_->reset();
  }

  cv::Mat cv_src, fgmask;
  cv_src = image_container::vital_to_ocv(image->get_image(),image_container::BGR_COLOR );

  d_->process_image(cv_src, fgmask);

  return std::make_shared<ocv::image_container>(fgmask,image_container::BGR_COLOR);
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
