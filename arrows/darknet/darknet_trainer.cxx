/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "darknet_trainer.h"
#include "darknet_custom_resize.h"

#include <vital/logger/logger.h>
#include <vital/util/cpu_timer.h>
#include <vital/vital_foreach.h>

#include <arrows/ocv/image_container.h>
#include <kwiversys/SystemTools.hxx>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include <string>
#include <sstream>
#include <fstream>

namespace kwiver {
namespace arrows {
namespace darknet {

// ==================================================================
class darknet_trainer::priv
{
public:
  priv()
    : m_net_config( "" )
    , m_seed_weights( "" )
    , m_output_weights( "" )
    , m_train_directory( "darknet_training" )
    , m_skip_format( false )
    , m_gpu_index( -1 )
    , m_resize_option( "maintain_ar" )
    , m_scale( 1.0 )
    , m_resize_i( 0 )
    , m_resize_j( 0 )
    , m_chip_step( 100 )
    , m_overlap_required( 0.05 )
    , m_chips_w_gt_only( false )
  {}

  ~priv()
  {}

  // Items from the config
  std::string m_net_config;
  std::string m_seed_weights;
  std::string m_output_weights;
  std::string m_train_directory;
  bool m_skip_format;
  int m_gpu_index;
  std::string m_resize_option;
  double m_scale;
  int m_resize_i;
  int m_resize_j;
  int m_chip_step;
  double m_overlap_required;
  bool m_chips_w_gt_only;

  // Helper functions
  std::vector< std::string > format_images( std::string folder,
    std::vector< std::string > image_names,
    std::vector< kwiver::vital::detected_object_set_sptr > groundtruth );

  bool print_detections(
    std::string filename,
    kwiver::vital::detected_object_set_sptr all_detections,
    kwiver::vital::bounding_box_d region );

  void generate_fn(
    std::string folder, std::string& gt,
    std::string& img, const int len = 10 );

  kwiver::vital::logger_handle_t m_logger;
};


// ==================================================================
darknet_trainer::
darknet_trainer()
  : d( new priv() )
{
}

darknet_trainer::
~darknet_trainer()
{
}


// --------------------------------------------------------------------
vital::config_block_sptr
darknet_trainer::
get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  config->set_value( "net_config", d->m_net_config,
    "Name of network config file." );
  config->set_value( "seed_weights", d->m_seed_weights,
    "Optional input seed weights file." );
  config->set_value( "output_weights", d->m_output_weights,
    "Output weights file." );
  config->set_value( "train_directory", d->m_train_directory,
    "Temp directory for all files used in training." );
  config->set_value( "skip_format", d->m_skip_format,
    "Skip file formatting, assume that the train_directory is pre-populated "
    "with all files required for model training." );
  config->set_value( "gpu_index", d->m_gpu_index,
    "GPU index. Only used when darknet is compiled with GPU support." );
  config->set_value( "resize_option", d->m_resize_option,
    "Pre-processing resize option, can be: disabled, maintain_ar, scale, "
    "chip, or chip_and_original." );
  config->set_value( "scale", d->m_scale,
    "Image scaling factor used when resize_option is scale or chip." );
  config->set_value( "resize_ni", d->m_resize_i,
    "Width resolution after resizing" );
  config->set_value( "resize_nj", d->m_resize_j,
    "Height resolution after resizing" );
  config->set_value( "chip_step", d->m_chip_step,
    "When in chip mode, the chip step size between chips." );
  config->set_value( "overlap_required", d->m_overlap_required,
    "Percentage of which a target must appear on a chip for it to be included "
    "as a training sample for said chip." );
  config->set_value( "chips_w_gt_only", d->m_chips_w_gt_only,
    "Only chips with valid groundtruth objects on them will be included in "
    "training." );

  return config;
}


// --------------------------------------------------------------------
void
darknet_trainer::
set_configuration( vital::config_block_sptr config_in )
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();

  config->merge_config( config_in );

  this->d->m_net_config  = config->get_value< std::string >( "net_config" );
  this->d->m_seed_weights = config->get_value< std::string >( "seed_weights" );
  this->d->m_output_weights = config->get_value< std::string >( "output_weights" );
  this->d->m_train_directory = config->get_value< std::string >( "train_directory" );
  this->d->m_skip_format = config->get_value< bool >( "skip_format" );
  this->d->m_gpu_index   = config->get_value< int >( "gpu_index" );
  this->d->m_resize_option = config->get_value< std::string >( "resize_option" );
  this->d->m_scale       = config->get_value< double >( "scale" );
  this->d->m_resize_i    = config->get_value< int >( "resize_ni" );
  this->d->m_resize_j    = config->get_value< int >( "resize_nj" );
  this->d->m_chip_step   = config->get_value< int >( "chip_step" );
  this->d->m_overlap_required = config->get_value< double >( "overlap_required" );
  this->d->m_chips_w_gt_only = config->get_value< bool >( "chips_w_gt_only" );
}


// --------------------------------------------------------------------
bool
darknet_trainer::
check_configuration( vital::config_block_sptr config ) const
{
  std::string net_config = config->get_value< std::string >( "net_config" );

  if( net_config.empty() || !kwiversys::SystemTools::FileExists( net_config ) )
  {
    LOG_ERROR( logger(), "net config file \"" << net_config << "\" not found." );
    return false;
  }

  return true;
}


// --------------------------------------------------------------------
void
darknet_trainer::
train_from_disk(std::vector< std::string > train_image_names,
  std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
  std::vector< std::string > test_image_names,
  std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth)
{
  // Format images correctly in tmp folder
  if( !d->m_skip_format )
  {
    // Delete and reset folder contents
    if( boost::filesystem::exists( d->m_train_directory ) &&
        boost::filesystem::is_directory( d->m_train_directory ) )
    {
      boost::filesystem::remove_all( d->m_train_directory );
    }

    boost::filesystem::path dir( d->m_train_directory );
    boost::filesystem::create_directories( dir );

    // Format train images
    std::vector< std::string > train_list, test_list;

    train_list = d->format_images( d->m_train_directory + "/train_images",
      train_image_names, train_groundtruth );
    test_list = d->format_images( d->m_train_directory + "/test_images",
      test_image_names, test_groundtruth );

    // Generate train/test image list and header information
    std::string python_cmd = "python -c '";
    std::string import_cmd = "import kwiver.arrows.darknet;";
    std::string header_cmd = "generate_header()";
    std::string end_quote  = "'";

    std::string full_cmd = python_cmd + " " + import_cmd + " " + header_cmd;

    system( full_cmd.c_str() );
  }

  // Run training routine
#ifdef WIN32
  std::string darknet_cmd = "darknet.exe";
#else
  std::string darknet_cmd = "darknet";
#endif
  std::string darknet_args = "-i " + boost::lexical_cast< std::string >( d->m_gpu_index ) +
    " detector train " + d->m_train_directory + "/YOLOv2.data " +
    d->m_net_config;

  if( !d->m_seed_weights.empty() )
  {
    darknet_args = darknet_args + " " + d->m_seed_weights;
  }

  std::string full_cmd = darknet_cmd + " " + darknet_args;

  system( full_cmd.c_str() );
}

// --------------------------------------------------------------------
std::vector< std::string >
darknet_trainer::priv::
format_images( std::string folder,
  std::vector< std::string > image_names,
  std::vector< kwiver::vital::detected_object_set_sptr > groundtruth )
{
  std::vector< std::string > output_fns;

  boost::filesystem::path dir( folder );
  boost::filesystem::create_directories( dir );

  for( unsigned fid = 0; fid < image_names.size(); ++fid )
  {
    const std::string image_fn = image_names[fid];
    kwiver::vital::detected_object_set_sptr detections_ptr = groundtruth[fid];
    kwiver::vital::detected_object_set_sptr scaled_detections_ptr = groundtruth[fid]->clone();

    // Scale and break up image according to settings
    cv::Mat original_image, resized_image;
    original_image = cv::imread( image_fn, -1 );

    if( original_image.rows == 0 || original_image.cols == 0 )
    {
      std::cout << "Could not load image " << image_fn << std::endl;
      return std::vector< std::string >();
    }

    double resized_scale = 1.0;

    if( m_resize_option != "disabled" )
    {
      resized_scale = format_image( original_image, resized_image,
        m_resize_option, m_scale, m_resize_i, m_resize_j );
      scaled_detections_ptr->scale( resized_scale );
    }
    else
    {
      resized_image = original_image;
      scaled_detections_ptr = detections_ptr;
    }

    if( m_resize_option != "chip" && m_resize_option != "chip_and_original" )
    {
      std::string gt_file, img_file;
      generate_fn( folder, gt_file, img_file );

      kwiver::vital::bounding_box_d roi_box( 0, 0, resized_image.cols, resized_image.rows );
      if( print_detections( gt_file, scaled_detections_ptr, roi_box ) )
      {
        cv::imwrite( img_file, resized_image );
      }
    }
    else
    {
      // Chip up and process scaled image
      for( int i = 0; i < resized_image.cols; i += m_chip_step )
      {
        int cw = i + m_resize_i;

        if( cw > resized_image.cols )
        {
          cw = resized_image.cols - i;
        }
        else
        {
          cw = m_resize_i;
        }

        for( int j = 0; j < resized_image.rows; j += m_chip_step )
        {
          int ch = j + m_resize_j;

          if( ch > resized_image.rows )
          {
            ch = resized_image.rows - j;
          }
          else
          {
            ch = m_resize_j;
          }

          cv::Mat cropped_image = resized_image( cv::Rect( i, j, cw, ch ) );
          cv::Mat resized_crop;

          scale_image_maintaining_ar( cropped_image,
            resized_crop, m_resize_i, m_resize_j );

          std::string gt_file, img_file;
          generate_fn( folder, gt_file, img_file );

          kwiver::vital::bounding_box_d roi_box( i, j, i + m_resize_i, j + m_resize_j );
          if( print_detections( gt_file, scaled_detections_ptr, roi_box ) )
          {
            cv::imwrite( img_file, resized_crop );
          }
        }
      }

      // Process full sized image if enabled
      if( m_resize_option == "chip_and_original" )
      {
        cv::Mat scaled_original;

        double scaled_original_scale = scale_image_maintaining_ar( original_image,
          scaled_original, m_resize_i, m_resize_j );

        kwiver::vital::detected_object_set_sptr scaled_original_dets_ptr = groundtruth[fid]->clone();
        scaled_original_dets_ptr->scale( scaled_original_scale );

        std::string gt_file, img_file;
        generate_fn( folder, gt_file, img_file );

        kwiver::vital::bounding_box_d roi_box( 0, 0,
          scaled_original.cols, scaled_original.rows );

        if( print_detections( gt_file, scaled_original_dets_ptr, roi_box ) )
        {
          cv::imwrite( img_file, scaled_original );
        }
      }
    }
  }

  return output_fns;
}

bool
darknet_trainer::priv::
print_detections(
  std::string filename,
  kwiver::vital::detected_object_set_sptr all_detections,
  kwiver::vital::bounding_box_d region )
{
  kwiver::vital::detected_object::vector_t input = all_detections->select();
  std::vector< std::string > to_write;

  const double width = region.width();
  const double height = region.height();

  VITAL_FOREACH( auto detection, input )
  {
    kwiver::vital::bounding_box_d det_box = detection->bounding_box();
    kwiver::vital::bounding_box_d overlap = kwiver::vital::intersection( region, det_box );

    if( det_box.area() > 0 &&
        overlap.max_x() > overlap.min_x() &&
        overlap.max_y() > overlap.min_y() &&
        overlap.area() / det_box.area() >= m_overlap_required )
    {
      std::string category = "1";
      //double tmp;
      //detection->type()->get_most_likely( category, tmp );

      double min_x = overlap.min_x() - region.min_x();
      double min_y = overlap.min_y() - region.min_y();

      double max_x = overlap.max_x() - region.min_x();
      double max_y = overlap.max_y() - region.min_y();

      std::string line = category + " ";

      line += boost::lexical_cast< std::string >( 0.5 * ( min_x + max_x ) / width ) + " ";
      line += boost::lexical_cast< std::string >( 0.5 * ( min_y + max_y ) / height ) + " ";

      line += boost::lexical_cast< std::string >( overlap.width() / width ) + " ";
      line += boost::lexical_cast< std::string >( overlap.height() / height );

      to_write.push_back( line );
    }
  }

  if( !m_chips_w_gt_only || !to_write.empty() )
  {
    std::ofstream fout( filename.c_str() );

    VITAL_FOREACH( std::string line, to_write )
    {
      fout << line << std::endl;
    }

    fout.close();
    return true;
  }

  return false;
}

void
darknet_trainer::priv::
generate_fn( std::string folder, std::string& gt, std::string& img, const int len )
{
  static const char alphanum[] =
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz";

  std::string s( len, ' ' );

  for( int i = 0; i < len; ++i )
  {
    s[i] = alphanum[ rand() % (sizeof(alphanum) - 1) ];
  }

  gt = folder + "/" + s + ".txt";
  img = folder + "/" + s + ".png";
}

} } } // end namespace
