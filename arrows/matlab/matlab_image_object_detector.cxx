/*ckwg +29
 * Copyright 2016-2018 by Kitware, Inc.
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
#include "matlab_util.h"

#include <kwiversys/SystemTools.hxx>

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>

#include <string>
#include <sstream>
#include <fstream>

namespace kwiver {
namespace arrows {
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
 *   - set_configuration() - accepts a new configuration into the detector. (?)
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
 *  - detected_object_set - array containing detections; boxes and confidence
 *  - detected_object_classification - array of structs containing the classification
 *    labels and scores.
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
    : m_first( true )
  {}

  ~priv()
  {}

  matlab_engine* engine()
  {
    // JIT allocation of engine if needed.
    //
    //@bug Because of the way these algorithms are managed and
    // duplicated, the matlab engine pointer must be shared with all
    // copies of this algorithm.  This is not optimal, since different
    // detectors may collide in the engine.  Doing the JIT creation
    // causes problems in that it allocates multiple engines. The real
    // solution is to get better control of creating these objects and
    // not have them clone themselves all the time.
    if ( ! m_matlab_engine)
    {
      m_matlab_engine.reset( new matlab_engine );
      LOG_DEBUG( m_logger, "Allocating a matlab engine @ " << m_matlab_engine );
    }

    return m_matlab_engine.get();
  }


  // ------------------------------------------------------------------
  void check_result()
  {
    const std::string& results( engine()->output() );
    if ( results.size() > 0 )
    {
      LOG_INFO( m_logger, engine() << " Matlab output: " << results );
    }
  }


  // ------------------------------------------------------------------
  void eval( const std::string& expr )
  {
    LOG_DEBUG( m_logger, engine() << " Matlab eval: " << expr );
    engine()->eval( expr );
    check_result();
  }


  // ------------------------------------------------------------------
  void initialize_once()
  {
    if ( ! m_first)
    {
      return;
    }

    m_first = false;

    std::ifstream t( m_matlab_program );
    std::stringstream buffer;
    buffer << t.rdbuf();
    eval( buffer.str() );

    // Create path to program file so we can do addpath('path');
    std::string full_path = kwiversys::SystemTools::CollapseFullPath( m_matlab_program );
    full_path = kwiversys::SystemTools::GetFilenamePath( full_path );

    eval( "addpath('" + full_path + "')" );

    // Get config values for this algorithm by extracting the subblock
    auto algo_config = m_config->subblock( "config" );

    // Iterate over all values in this config block and pass the values
    // to the matlab as variable assignments.
    auto keys = algo_config->available_values();
    for( auto k : keys )
    {
      std::stringstream config_command;
      config_command <<  k << "=" << algo_config->get_value<std::string>( k ) << ";";
      eval( config_command.str() );
    }// end foreach

    eval( "detector_initialize()" );
  }


  // --- instance data -----
  kwiver::vital::logger_handle_t m_logger;
  bool m_first;

  // MatLab wrapper parameters
  std::string m_matlab_program;       // name of matlab program
  vital::config_block_sptr m_config;

private:
  // MatLab support. The engine is allocated at the latest time.
  std::shared_ptr<matlab_engine> m_matlab_engine;

}; // end class matlab_image_object_detector::priv


// ==================================================================

matlab_image_object_detector::
matlab_image_object_detector()
  : d( new priv )
{
  attach_logger( "arrows.matlab.matlab_image_object_detector" );
  d->m_logger = logger();
}


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
  d->m_config = config;

  // Load specified program file into matlab engine
  d->m_matlab_program = config->get_value<std::string>( "program_file" );
}


// ------------------------------------------------------------------
bool
matlab_image_object_detector::
check_configuration(vital::config_block_sptr config) const
{
  // d->eval( "check_configuration()" );

  //+ not sure this has any value.
  // Need to get a return value back.
  // Could execute "retval = check_configuration()"
  // and them retrieve the results
  //@todo  check output buffer for message to throw

  return true;
}


// ------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
matlab_image_object_detector::
detect( kwiver::vital::image_container_sptr image_data) const
{
  d->initialize_once();

  auto detected_set = std::make_shared< kwiver::vital::detected_object_set>();

  // convert image container to matlab image
  MxArraySptr mx_image = convert_mx_image( image_data );

  d->engine()->put_variable( "in_image", mx_image );
  d->eval( "detect(in_image)" );

  MxArraySptr detections = d->engine()-> get_variable( "detected_object_set" ); // throws
  d->check_result();

  // Check dimensionality of returned array
  size_t col = detections->cols();
  if ( col < 5 )
  {
    throw std::runtime_error ( "Insufficient columns in detections array. Must have 5 columns." );
  }

  if (col > 5)
  {
    LOG_WARN( d->m_logger, "Extra columns in detections array. Ignored." );
  }

  size_t num_det = detections->rows();

  // Get the classification info if there
  // TBD catch exception (if any) and mark as no classifications
  size_t class_rows( 0 );
  size_t class_cols( 0 );
  MxArraySptr class_dims;
  d->eval( "temp_temp=size(detected_object_classification);" );
  try
  {
    class_dims = d->engine()-> get_variable( "temp_temp" ); // throws
    class_rows = class_dims->at<size_t>(0);
    class_cols = class_dims->at<size_t>(1);
  }
  catch( ... ) { }

  // Get the mask info if there is any
  size_t mask_entries( 0 );
  MxArraySptr mask_dims;
  d->eval( "temp_temp_temp=size(detected_object_chips);" );
  try
  {
    mask_dims = d->engine()-> get_variable( "temp_temp_temp" ); // throws
    mask_entries = mask_dims->at<size_t>(0);\
  }
  catch( ... ) { }

  // Process each detection and create an object
  for ( size_t i = 0; i < num_det; i++ )
  {
    kwiver::vital::bounding_box_d bbox( detections->at<double>(i, (size_t) 0), // tl-x
                                        detections->at<double>(i, (size_t) 1), // tl-y
                                        detections->at<double>(i, (size_t) 2), // lr-x
                                        detections->at<double>(i, (size_t) 3) ); // lr-y

    // Save classifications in DOT
    kwiver::vital::detected_object_type_sptr dot;
    if ( class_rows ) // there are some classification details
    {
      dot = std::make_shared< kwiver::vital::detected_object_type >();

      for ( size_t cc = 0; cc < class_cols; ++cc )
      {
        // Extract name and score from matlab
        std::stringstream cmd;
        cmd << "temp_temp=detected_object_classification(" << (i+1) << "," << (cc+1) << ").name;";
        d->eval( cmd.str() );
        MxArraySptr temp = d->engine()-> get_variable( "temp_temp" );

        // If the name is empty, then there are no more names for this detection.
        if ( temp->size() == 0 )
        {
          continue;
        }

        const std::string c_name = temp->getString();

        cmd.str(""); // reset command string
        cmd << "temp_temp=detected_object_classification(" << (i+1) << "," << (cc+1) << ").score;";
        d->eval( cmd.str() );
        temp =  d->engine()-> get_variable( "temp_temp" );
        const double c_score = temp->at<double>(0);

        dot->set_score( c_name, c_score );
      } // end for cc
    }

    // Save mask in DOT
    auto detection = std::make_shared< kwiver::vital::detected_object >( bbox, detections->at<double>(i, 4), dot );

    if( mask_entries )
    {
      kwiver::vital::image_container_sptr mask;

      // Extract name and score from matlab
      std::stringstream cmd;
      cmd << "temp_temp_temp=detected_object_chip(" << (i+1) << ").chip;";
      d->eval( cmd.str() );
      MxArraySptr temp = d->engine()-> get_variable( "temp_temp_temp" );

      // If the name is empty, then there are no more names for this detection.
      if ( temp->size() == 0 )
      {
        continue;
      }

      mask = convert_mx_image( temp );

      detection->set_mask( mask );
    }

    d->eval( "clear temp_temp;" );
    d->eval( "clear temp_temp_temp;" );

    // Add this detection to the set
    detected_set->add( detection );

  } // end for

  return detected_set;
}

} } } // end namespace
