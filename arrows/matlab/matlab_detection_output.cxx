/*ckwg +29
 * Copyright 2016-2018, 2020 by Kitware, Inc.
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
 * \brief Implementation of matlab image object output
 */

#include "matlab_detection_output.h"
#include "matlab_engine.h"
#include "matlab_util.h"

#include <kwiversys/SystemTools.hxx>

#include <string>

namespace kwiver {
namespace arrows {
namespace matlab {

// ------------------------------------------------------------------
class matlab_detection_output::priv
{
public:
  priv( matlab_detection_output* parent)
    : m_parent( parent )
    , m_first( true )
  { }

  ~priv() { }

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
  matlab_detection_output* m_parent;
  kwiver::vital::logger_handle_t m_logger;
  bool m_first;

  // MatLab wrapper parameters
  std::string m_matlab_program;       // name of matlab program
  vital::config_block_sptr m_config;

private:
  // MatLab support. The engine is allocated at the latest time.
  std::shared_ptr<matlab_engine> m_matlab_engine;

};


// ==================================================================
matlab_detection_output::
matlab_detection_output()
  : d( new matlab_detection_output::priv( this ) )
{
}


matlab_detection_output::
~matlab_detection_output()
{
}


// ------------------------------------------------------------------
vital::config_block_sptr
matlab_detection_output::
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
matlab_detection_output::
set_configuration( vital::config_block_sptr config )
{
  d->m_config = config;

  // Load specified program file into matlab engine
  d->m_matlab_program = config->get_value<std::string>( "program_file" );
}


// ------------------------------------------------------------------
bool
matlab_detection_output::
check_configuration( vital::config_block_sptr config ) const
{
  return true;
}


// ------------------------------------------------------------------
void
matlab_detection_output::
write_set( const kwiver::vital::detected_object_set_sptr detections,
           std::string const&                            image_name )
{
  d->initialize_once();

  MxArraySptr mx_image_name = std::make_shared<MxArray>( MxArray::Cell() );
  mx_image_name->set( 0, image_name );
  MxArraySptr mx_detections = std::make_shared<MxArray>();
  MxArraySptr mx_class =  std::make_shared<MxArray>( MxArray::Cell(detections->size(), 2) );
  unsigned det_index(0);

  for( const auto det : *detections )
  {
    const kwiver::vital::bounding_box_d bbox( det->bounding_box() );

    // add each detection to matlab array
    mx_detections->set( det_index, 0, bbox.min_x() );
    mx_detections->set( det_index, 1, bbox.min_y() );
    mx_detections->set( det_index, 2, bbox.max_x() );
    mx_detections->set( det_index, 3, bbox.max_y() );
    mx_detections->set( det_index, 4, det->confidence() );

    // Process classifications if there are any
    const auto dot( det->type() );
    if ( dot )
    {
      const auto name_list( dot->class_names() );
      for( auto name : name_list )
      {
        // Add classification entry to cell array
        mx_class->set( det_index, 0, name.c_str() );
        mx_class->set( det_index, 1, dot->score(name) );
      } // end foreach
    }

    ++det_index; // increment array index
  } // end foreach

  // push detection set information into matlab
  d->engine()->put_variable(  "detected_object_set" , mx_detections );
  d->engine()->put_variable(  "detected_object_classification" , mx_class );
  d->engine()->put_variable(  "image_name" , mx_image_name );

  // call matlab function
  d->eval( "write_set(detected_object_set, detected_object_classification, image_name)" );
}

} } } // end namespace
