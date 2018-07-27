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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

#include "create_detection_grid_process.h"

#include <kwiver_type_traits.h>

#include <vital/types/bounding_box.h>
#include <vital/types/detected_object_set.h>

#include <sprokit/pipeline/process_exception.h>

namespace kwiver
{

create_detection_grid_process
::create_detection_grid_process( vital::config_block_sptr const& config )
  : process( config ), m_width(1), m_height(1)
{
  // Attach our logger name to process logger
  attach_logger( vital::get_logger( name() ) );

  make_ports();
  make_config();
}

create_detection_grid_process
::~create_detection_grid_process()
{
}

void create_detection_grid_process
::_configure()
{

  vital::config_block_sptr algo_config = get_config();

  if(!algo_config->has_value("detection_width"))
  {
    throw sprokit::invalid_configuration_exception(
      name(), "missing required parameter detection_width");
  }
  if(!algo_config->has_value("detection_height"))
  {
    throw sprokit::invalid_configuration_exception(
      name(), "missing required parameter detection_height");
  }

  m_width = algo_config->get_value<size_t>("detection_width");
  m_height = algo_config->get_value<size_t>("detection_height");

  if(m_width <= 0 || m_height <= 0)
  {
    throw sprokit::invalid_configuration_exception(
      name(), "Width and height must be positive values");
  }

}

void
create_detection_grid_process
::_step()
{

  vital::detected_object_set_sptr grid(new vital::detected_object_set());
  vital::image_container_sptr image = grab_from_port_using_trait(image);
  size_t img_width = image->width();
  size_t img_height = image->height();

  if(m_width > img_width || m_height > img_height)
  {
    throw sprokit::invalid_configuration_exception(
      name(), "Detection width and height must be no more than image width and height");
  }

  // Get any non-overlapping grid spaces
  // Note that the last column and row are missing here
  for(int i = 0; i + m_width < img_width ; i+=m_width)
  {
    for(int j = 0; j + m_height < img_height ; j+=m_height)
    {
      vital::bounding_box<double> bbox(i, j, i+m_width-1, j+m_height-1);
      vital::detected_object_sptr det_obj(new vital::detected_object(bbox));
      grid->add(det_obj);
    }
  }

  // Now get the bottom row
  for (int i = 0; i + m_width < img_width ; i+=m_width)
  {
    vital::bounding_box<double> bbox(i, img_height - m_height, i+m_width-1, img_height-1);
    vital::detected_object_sptr det_obj(new vital::detected_object(bbox));
    grid->add(det_obj);
  }

  // Now get the bottom column
  for (int j = 0; j + m_height < img_height ; j+=m_height)
  {
    vital::bounding_box<double> bbox(img_width - m_width, j , img_width-1, j+m_height-1);
    vital::detected_object_sptr det_obj(new vital::detected_object(bbox));
    grid->add(det_obj);
  }

  // Our last special case, the bottom right
  vital::bounding_box<double> bbox(img_width - m_width, img_height - m_height , img_width-1, img_height-1);
  vital::detected_object_sptr det_obj(new vital::detected_object(bbox));
  grid->add(det_obj);

  push_to_port_using_trait(detected_object_set, grid);
}

void
create_detection_grid_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait(image, required);

  // -- output --
  declare_output_port_using_trait(detected_object_set, required);
}

void
create_detection_grid_process
::make_config()
{

}

} // end namespace
