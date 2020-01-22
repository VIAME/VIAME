/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

/**
 * \file
 * \brief Implementation of create_detection_grid.h
 */

#include "create_detection_grid.h"

#include <vital/types/bounding_box.h>
#include <vital/types/detected_object_set.h>

#include <vital/exceptions/algorithm.h>

namespace kwiver {
namespace arrows {
namespace core {

using namespace kwiver::vital;


/// Private implementation class
class create_detection_grid::priv
{
public:
  /// Constructor
  priv()
    : width(0)
    , height(0)
    , x_step(0)
    , y_step(0)
    , m_logger( vital::get_logger(
        "arrows.core.create_detection_grid" ) )
  {
  }

  double width, height, x_step, y_step;

  /// Logger handle
  vital::logger_handle_t m_logger;
};

/// Constructor
create_detection_grid
::create_detection_grid()
  : d_( new priv )
{
}


/// Destructor
create_detection_grid
::~create_detection_grid() noexcept
{
}


/// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
create_detection_grid
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "detection_width", d_->width,
    "Width of each detection in the output grid." );
  config->set_value( "detection_height", d_->height,
    "Height of each detection in the output grid." );
  config->set_value( "x_step", d_->x_step,
    "How far apart along the x axis each detection is." );
  config->set_value( "y_step", d_->y_step,
    "How far apart along the y axis each detection is." );

  return config;
}


/// Set this algo's properties via a config block
void
create_detection_grid
::set_configuration( vital::config_block_sptr in_config )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  d_->width = config->get_value<double>( "detection_width" );
  d_->height = config->get_value<double>( "detection_height" );

  d_->x_step = config->get_value<double>( "x_step" );
  d_->y_step = config->get_value<double>( "y_step" );
}


bool
create_detection_grid
::check_configuration(vital::config_block_sptr config) const
{
  if(config->get_value<double>("detection_width") <= 0 || config->get_value<double>("detection_width") <= 0)
  {
    LOG_ERROR(d_->m_logger, "Detection width and height must be positive values");
    return false;
  }
  if(config->get_value<double>("x_step") <= 0 && config->get_value<double>("y_step") <= 0)
  {
    LOG_ERROR(d_->m_logger, "Detection steps must be positive values");
    return false;
  }
  return true;
}


vital::detected_object_set_sptr
create_detection_grid::
detect( vital::image_container_sptr image_data) const
{
  vital::detected_object_set_sptr grid(new vital::detected_object_set());
  const size_t img_width = image_data->width();
  const size_t img_height = image_data->height();

  if(d_->width > img_width || d_->height > img_height)
  {
    VITAL_THROW( vital::algorithm_configuration_exception,
      type_name(), impl_name(), "Detection width and height must be no more than image width and height");
  }

  // Get any non-overlapping grid spaces
  // Note that the last column and row are missing here
  for(int i = 0; i + d_->width < img_width ; i+=d_->x_step)
  {
    for(int j = 0; j + d_->height < img_height ; j+=d_->y_step)
    {
      vital::bounding_box<double> bbox(i, j, i+d_->width-1, j+d_->height-1);
      vital::detected_object_sptr det_obj(new vital::detected_object(bbox));
      grid->add(det_obj);
    }
  }

  // Now get the bottom row
  for (int i = 0; i + d_->width < img_width ; i+=d_->x_step)
  {
    vital::bounding_box<double> bbox(i, img_height - d_->height, i+d_->width-1, img_height-1);
    vital::detected_object_sptr det_obj(new vital::detected_object(bbox));
    grid->add(det_obj);
  }

  // Now get the bottom column
  for (int j = 0; j + d_->height < img_height ; j+=d_->y_step)
  {
    vital::bounding_box<double> bbox(img_width - d_->width, j , img_width-1, j+d_->height-1);
    vital::detected_object_sptr det_obj(new vital::detected_object(bbox));
    grid->add(det_obj);
  }

  // Our last special case, the bottom right
  vital::bounding_box<double> bbox(img_width - d_->width, img_height - d_->height , img_width-1, img_height-1);
  vital::detected_object_sptr det_obj(new vital::detected_object(bbox));
  grid->add(det_obj);

  return grid;

}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
