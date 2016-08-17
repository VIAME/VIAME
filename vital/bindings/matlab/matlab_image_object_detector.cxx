/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * \brief Implementation of matlab image object detector
 */

#include "matlab_image_object_detector.h"
#include "matlab_engine.h"

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>

namespace kwiver {
namespace vital {
namespace matlab {

// ----------------------------------------------------------------
/**
 * @class matlab_image_object_detector
 *
 * @brief Wrapper for matlab image detectors.
 *
 * This class represents a wrapper for image object detectors written
 * in MatLab.
 *
 * Image object detectors written in MatLab must support the following
 * interface, at a minimum.
 *
 * Functions:
 *   - impl_name() - returns the implementation name for the matlab algorithm
 *
 *   - get_configuration() - returns the required configuration (format to be determined)
 *     May just punt and pass a filename to the algorithm and let it decode the config.
 *
 *   - set_configuration() - accepts a new configuration into the detector.
 *
 *   - check_configuration() - returns error if there is a configuration problem
 *
 *   - detect() - performs detection operation using input variables as input and
 *     produces output on output variables.
 *
 * Input variables:
 *  - in_image - contains the input image. Shape of the array is the size of the image.
 *
 * Output variables:
 *  - detector_status - status code. 0 for operation completed successfully.
 *    Any other value indicates and error and the error message is returned in
 *    detector_status_msg
 *
 *  - detector_status_msg - Text describing the status/failure of the last API call.
 *
 */

// ----------------------------------------------------------------
/**
 * @brief
 *
 */
class matlab_image_object_detector::priv
{
public:
  // -- CONSTRUCTORS --
  priv()
    : m_matlab_engine( new matlab_engine )
  {}

  priv( const priv & other ) // copy ctor
    : m_matlab_engine( new matlab_engine )
    , m_matlab_program( other.m_matlab_program )
  {}

  ~priv()
  {}

  // MatLab support.
  const std::unique_ptr<matlab_engine> m_matlab_engine;

  // MatLab wrapper parameters
  std::string m_matlab_program;       // name of matlab program

}; // end class matlab_image_object_detector::priv


// ==================================================================

matlab_image_object_detector::
matlab_image_object_detector()
  : d( new priv )
{ }


matlab_image_object_detector::
matlab_image_object_detector( const matlab_image_object_detector& other)
  : d( new priv( *other.d ) )
{ }


 matlab_image_object_detector::
~matlab_image_object_detector()
{ }


// ------------------------------------------------------------------
vital::config_block_sptr
matlab_image_object_detector::
get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  config->set_value( "program_file", d->m_matlab_program,
                     "File name of the matlab image object detector program to run." );

  return config;
}


// ------------------------------------------------------------------
void
matlab_image_object_detector::
set_configuration(vital::config_block_sptr config)
{
  d->m_matlab_program = config->get_value<double>( "program_file" );

  d->m_matlab_engine->eval( "load " + d->m_matlab_program );
  //@todo check return code

  d->m_matlab_engine->eval( "initialize()" );

  // Convert configuration to matlab format
  // TBD
  // Since the config is a key/value pair, send in two pairs of strings?
  // As an array? How about repeated eval calls to a function set_config( key, value ).
  //

  // Pass configuration to matlab detector
  d->m_matlab_engine->eval( "set_configuration()" );
  //@todo check return code

}


// ------------------------------------------------------------------
bool
matlab_image_object_detector::
check_configuration(vital::config_block_sptr config) const
{
    d->m_matlab_engine->eval( "check_configuration()" );
  //@todo check return code

  //@todo  check output buffer for message to throw

  return true;
}


// ------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
matlab_image_object_detector::
detect( vital::image_container_sptr image_data) const
{
  auto detected_set = std::make_shared< kwiver::vital::detected_object_set>();

  // Ideally use this call
  // mxArraySptr mx_image = convert_image( image_data );

  cv::Mat src = kwiver::arrows::ocv::image_container::vital_to_ocv( image_data->get_image() );

  // convert image container to matlab image TBD
  mxArraySptr mx_image = create_mxByteArray( image_data->height(), image_data->width() );
  d->m_matlab_engine->put_variable( "in_image", mx_image );


  d->m_matlab_engine->eval( "detect()" );
  //@todo check return code

#if 0
  // process results
  for ( size_t i = 0; i < circles.size(); ++i )
  {
    // Center point [circles[i][0], circles[i][1]]
    // Radius circles[i][2]

    // Bounding box is center +/- radius
    kwiver::vital::bounding_box_d bbox( circles[i][0] - circles[i][2], circles[i][1] - circles[i][2],
                                        circles[i][0] + circles[i][2], circles[i][1] + circles[i][2] );

    auto dot = std::make_shared< kwiver::vital::detected_object_type >();
    dot->set_score( "circle", 1.0 );

    detected_set->add( std::make_shared< kwiver::vital::detected_object >( bbox, 1.0, dot ) );
  } // end for
#endif

  return detected_set;
}

} } } // end namespace
