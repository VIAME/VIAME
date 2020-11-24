// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "darknet_trainer.h"
#include "darknet_custom_resize.h"

#include <vital/util/cpu_timer.h>
#include <vital/types/detected_object_set_util.h>
#include <vital/vital_config.h>

#include <arrows/ocv/image_container.h>
#include <kwiversys/SystemTools.hxx>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>

namespace kwiver {
namespace arrows {
namespace darknet {

// =============================================================================
class darknet_trainer::priv
{
public:
  priv()
    : m_net_config( "" )
    , m_seed_weights( "" )
    , m_output_weights( "" )
    , m_train_directory( "darknet_training" )
    , m_skip_format( false )
    , m_gpu_index( 0 )
    , m_resize_option( "maintain_ar" )
    , m_scale( 1.0 )
    , m_resize_i( 0 )
    , m_resize_j( 0 )
    , m_chip_step( 100 )
    , m_overlap_required( 0.05 )
    , m_random_int_shift( 0.00 )
    , m_chips_w_gt_only( false )
    , m_crop_left( false )
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
  double m_random_int_shift;
  bool m_chips_w_gt_only;
  bool m_crop_left;

  // Helper functions
  std::vector< std::string > format_images( std::string folder,
    std::string prefix,
    std::vector< std::string > image_names,
    std::vector< kwiver::vital::detected_object_set_sptr > groundtruth,
    vital::category_hierarchy_sptr object_labels );

  bool print_detections(
    std::string filename,
    kwiver::vital::detected_object_set_sptr all_detections,
    kwiver::vital::bounding_box_d region,
    vital::category_hierarchy_sptr object_labels );

  void generate_fn(
    std::string image_folder, std::string gt_folder,
    std::string& image, std::string& gt, const int len = 10 );

  void save_chip(
    std::string filename,
    cv::Mat image );

  kwiver::vital::logger_handle_t m_logger;
};

// =============================================================================
darknet_trainer::
darknet_trainer()
  : d( new priv() )
{
}

darknet_trainer::
~darknet_trainer()
{
}

// -----------------------------------------------------------------------------
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
  config->set_value( "random_int_shift", d->m_random_int_shift,
    "Random intensity shift to add to each extracted chip [0.0,1.0]." );
  config->set_value( "chips_w_gt_only", d->m_chips_w_gt_only,
    "Only chips with valid groundtruth objects on them will be included in "
    "training." );
  config->set_value( "crop_left", d->m_crop_left,
    "Crop out the left portion of imagery, only using the left side." );

  return config;
}

// -----------------------------------------------------------------------------
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
  this->d->m_train_directory = config->get_value< std::string >( "train_directory" );
  this->d->m_skip_format = config->get_value< bool >( "skip_format" );
  this->d->m_gpu_index   = config->get_value< int >( "gpu_index" );
  this->d->m_resize_option = config->get_value< std::string >( "resize_option" );
  this->d->m_scale       = config->get_value< double >( "scale" );
  this->d->m_resize_i    = config->get_value< int >( "resize_ni" );
  this->d->m_resize_j    = config->get_value< int >( "resize_nj" );
  this->d->m_chip_step   = config->get_value< int >( "chip_step" );
  this->d->m_overlap_required = config->get_value< double >( "overlap_required" );
  this->d->m_random_int_shift = config->get_value< double >( "random_int_shift" );
  this->d->m_chips_w_gt_only = config->get_value< bool >( "chips_w_gt_only" );
  this->d->m_crop_left   = config->get_value< bool >( "crop_left" );
}

// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
void
darknet_trainer::
train_from_disk(
  vital::category_hierarchy_sptr object_labels,
  std::vector< std::string > train_image_names,
  std::vector< kwiver::vital::detected_object_set_sptr > train_groundtruth,
  std::vector< std::string > test_image_names,
  std::vector< kwiver::vital::detected_object_set_sptr > test_groundtruth)
{
  // Format images correctly in tmp folder
  if( !d->m_skip_format )
  {
    // Delete and reset folder contents
    if( kwiversys::SystemTools::FileExists( d->m_train_directory, false ) &&
        kwiversys::SystemTools::FileIsDirectory( d->m_train_directory ) )
    {
      kwiversys::SystemTools::RemoveADirectory( d->m_train_directory );
    }

    kwiversys::SystemTools::MakeDirectory( d->m_train_directory );

    // Format train images
    std::vector< std::string > train_list, test_list;

    train_list = d->format_images( d->m_train_directory, "train",
      train_image_names, train_groundtruth, object_labels );
    test_list = d->format_images( d->m_train_directory, "test",
      test_image_names, test_groundtruth, object_labels );

    // Generate train/test image list and header information
#ifdef WIN32
    std::string python_cmd = "python.exe -c '";
#else
    std::string python_cmd = "python -c '";
#endif
    std::string import_cmd = "import kwiver.arrows.darknet.generate_headers as dth;";
    std::string header_cmd = "dth.generate_yolo_v2_headers(";

    std::string header_args = "\"" + d->m_train_directory + "\",[";
    for( auto label : object_labels->child_class_names() )
    {
      header_args = header_args + "\"" + label + "\",";
    }
    header_args = header_args +"]," + std::to_string( d->m_resize_i );
    header_args = header_args + "," + std::to_string( d->m_resize_j );
    header_args = header_args + ",\"" + d->m_net_config + "\"";

    std::string header_end  = ")'";

    std::string full_cmd = python_cmd + import_cmd + header_cmd + header_args + header_end;

    if ( system( full_cmd.c_str() ) != 0 )
    {
      LOG_WARN( logger(), "System call \"" << full_cmd << "\" failed" );
    }
  }

  // Run training routine
#ifdef WIN32
  std::string darknet_cmd = "darknet.exe";
#else
  std::string darknet_cmd = "darknet";
#endif
  std::string darknet_args = "-i " + std::to_string( d->m_gpu_index ) +
    " detector train " + d->m_train_directory + "/yolo_v2.data "
                       + d->m_train_directory + "/yolo_v2.cfg ";

  if( !d->m_seed_weights.empty() )
  {
    darknet_args = darknet_args + " " + d->m_seed_weights;
  }

  std::string full_cmd = darknet_cmd + " " + darknet_args;

  std::cout << "Running " << full_cmd << std::endl;

  if ( system( full_cmd.c_str() ) != 0 )
  {
    LOG_WARN( logger(), "System call \"" << full_cmd << "\" failed" );
  }
}

// -----------------------------------------------------------------------------
std::vector< std::string >
darknet_trainer::priv::
format_images( std::string folder, std::string prefix,
  std::vector< std::string > image_names,
  std::vector< kwiver::vital::detected_object_set_sptr > groundtruth,
  vital::category_hierarchy_sptr object_labels )
{
  std::vector< std::string > output_fns;

  std::string image_folder = folder + "/" + prefix + "_images";
  std::string label_folder = folder + "/" + prefix + "_labels";

  kwiversys::SystemTools::MakeDirectory( image_folder );
  kwiversys::SystemTools::MakeDirectory( label_folder );

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

    if( m_crop_left )
    {
      original_image = cv::Mat( original_image,
        cv::Rect( 0, 0, original_image.cols/2, original_image.rows ) );
    }

    double resized_scale = 1.0;

    if( m_resize_option != "disabled" )
    {
      resized_scale = format_image( original_image, resized_image,
        m_resize_option, m_scale, m_resize_i, m_resize_j );
      scale_detections( scaled_detections_ptr, resized_scale );
    }
    else
    {
      resized_image = original_image;
      scaled_detections_ptr = detections_ptr;
    }

    if( m_resize_option != "chip" && m_resize_option != "chip_and_original" )
    {
      std::string img_file, gt_file;
      generate_fn( image_folder, label_folder, img_file, gt_file );

      kwiver::vital::bounding_box_d roi_box( 0, 0, resized_image.cols, resized_image.rows );
      if( print_detections( gt_file, scaled_detections_ptr, roi_box, object_labels ) )
      {
        save_chip( img_file, resized_image );
      }
    }
    else
    {
      // Chip up and process scaled image
      for( int i = 0; i < resized_image.cols - m_resize_i + m_chip_step; i += m_chip_step )
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

        for( int j = 0; j < resized_image.rows - m_resize_j + m_chip_step; j += m_chip_step )
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

          std::string img_file, gt_file;
          generate_fn( image_folder, label_folder, img_file, gt_file );

          kwiver::vital::bounding_box_d roi_box( i, j, i + m_resize_i, j + m_resize_j );
          if( print_detections( gt_file, scaled_detections_ptr, roi_box, object_labels ) )
          {
            save_chip( img_file, resized_crop );
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
        scale_detections( scaled_original_dets_ptr, scaled_original_scale );

        std::string img_file, gt_file;
        generate_fn( image_folder, label_folder, img_file, gt_file );

        kwiver::vital::bounding_box_d roi_box( 0, 0,
          scaled_original.cols, scaled_original.rows );

        if( print_detections( gt_file, scaled_original_dets_ptr, roi_box, object_labels ) )
        {
          save_chip( img_file, scaled_original );
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
  kwiver::vital::bounding_box_d region,
  vital::category_hierarchy_sptr object_labels )
{
  std::vector< std::string > to_write;

  const double width = region.width();
  const double height = region.height();

  auto ie = all_detections->cend();
  for ( auto detection = all_detections->cbegin(); detection != ie; ++detection )
  {
    kwiver::vital::bounding_box_d det_box = (*detection)->bounding_box();
    kwiver::vital::bounding_box_d overlap = kwiver::vital::intersection( region, det_box );

    if( det_box.area() > 0 &&
        overlap.max_x() > overlap.min_x() &&
        overlap.max_y() > overlap.min_y() &&
        overlap.area() / det_box.area() >= m_overlap_required )
    {
      std::string category;

      if( !(*detection)->type() )
      {
        std::cout << "Error: Detection missing type" << std::endl;
        return false;
      }

      (*detection)->type()->get_most_likely( category );

      if( object_labels->has_class_name( category ) )
      {
        category = std::to_string( object_labels->get_class_id( category ) );
      }
      else
      {
        std::cout << "Warning: Ignoring unlisted class " << category << std::endl;
        continue;
      }

      double min_x = overlap.min_x() - region.min_x();
      double min_y = overlap.min_y() - region.min_y();

      double max_x = overlap.max_x() - region.min_x();
      double max_y = overlap.max_y() - region.min_y();

      std::string line = category + " ";

      line += std::to_string( 0.5 * ( min_x + max_x ) / width ) + " ";
      line += std::to_string( 0.5 * ( min_y + max_y ) / height ) + " ";

      line += std::to_string( overlap.width() / width ) + " ";
      line += std::to_string( overlap.height() / height );

      to_write.push_back( line );
    }
  }

  if( !m_chips_w_gt_only || !to_write.empty() )
  {
    std::ofstream fout( filename.c_str() );

    for( std::string line : to_write )
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
generate_fn( std::string image_folder, std::string gt_folder,
             std::string& image, std::string& gt,
             VITAL_UNUSED const int len )
{
  static int sample_counter = 0;

  sample_counter++;

  std::ostringstream ss;
  ss << std::setw( 9 ) << std::setfill( '0' ) << sample_counter;
  std::string s = ss.str();

  image = image_folder + "/" + s + ".png";
  gt = gt_folder + "/" + s + ".txt";
}

void
darknet_trainer::priv::
save_chip( std::string filename, cv::Mat image )
{
  if( m_random_int_shift > 0.0 )
  {
    double rand_uniform = rand() / ( RAND_MAX + 1.0 );
    double start = ( 1.0 - m_random_int_shift );

    double sf = start + 2 * m_random_int_shift * rand_uniform;

    cv::Mat scaled_image = image * sf;
    cv::imwrite( filename, scaled_image );
  }
  else
  {
    cv::imwrite( filename, image );
  }
}

} } } // end namespace
