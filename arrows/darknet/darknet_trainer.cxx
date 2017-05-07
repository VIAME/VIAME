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

#include <arrows/ocv/image_container.h>
#include <kwiversys/SystemTools.hxx>

#include <opencv2/core/core.hpp>

#include <string>
#include <sstream>

namespace kwiver {
namespace arrows {
namespace darknet {

// ==================================================================
class darknet_trainer::priv
{
public:
  priv()
    : m_skip_format( false )
    , m_gpu_index( -1 )
    , m_resize_option( "maintain_ar" )
    , m_scale( 1.0 )
    , m_resize_i( 0 )
    , m_resize_j( 0 )
    , m_chip_step( 100 )
  {}

  ~priv()
  {}

  // Items from the config
  std::string m_net_config;
  std::string m_output_weights;
  std::string m_tmp_directory;
  bool m_skip_format;
  int m_gpu_index;
  std::string m_resize_option;
  double m_scale;
  int m_resize_i;
  int m_resize_j;
  int m_chip_step;

  // Helper functions
  double format_image( const cv::Mat& src, cv::Mat& dst );
  double scale_image_maintaining_ar( const cv::Mat& src, cv::Mat& dst );

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
  config->set_value( "output_weights", d->m_output_weights,
    "Output weights file." );
  config->set_value( "tmp_directory", d->m_tmp_directory,
    "Temp directory for all files used in training." );
  config->set_value( "skip_format", d->m_skip_format,
    "Skip file formatting, assume that the tmp_directory is pre-populated "
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
  this->d->m_output_weights = config->get_value< std::string >( "output_weights" );
  this->d->m_tmp_directory = config->get_value< std::string >( "tmp_directory" );
  this->d->m_skip_format = config->get_value< bool >( "skip_format" );
  this->d->m_gpu_index   = config->get_value< int >( "gpu_index" );
  this->d->m_resize_option = config->get_value< std::string >( "resize_option" );
  this->d->m_scale       = config->get_value< double >( "scale" );
  this->d->m_resize_i    = config->get_value< int >( "resize_i" );
  this->d->m_resize_j    = config->get_value< int >( "resize_j" );
  this->d->m_chip_step   = config->get_value< int >( "chip_step" );
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

}


} } } // end namespace
