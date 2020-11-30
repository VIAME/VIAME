// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of matlab image object filter
 */

#include "matlab_image_filter.h"
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
 * @class matlab_image_filter
 *
 * @brief Wrapper for matlab image filters.
 *
 * This class represents a wrapper for image object filters written
 * in MatLab.
 *
 * Image object filters written in MatLab must support the following
 * interface, at a minimum.
 *
 * Functions:
 *   - impl_name() - returns the implementation name for the matlab algorithm
 *
 *   - get_configuration() - returns the required configuration (format to be determined)
 *     May just punt and pass a filename to the algorithm and let it decode the config.
 *
 *   - set_configuration() - accepts a new configuration into the filter. (?)
 *
 *   - check_configuration() - returns error if there is a configuration problem
 *
 *   - filter() - performs detection operation using input variables as input and
 *     produces output on output variables.
 *
 */

// ----------------------------------------------------------------
/**
 * @brief
 *
 */
class matlab_image_filter::priv
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
    // filters may collide in the engine.  Doing the JIT creation
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

    eval( "filter_initialize()" );
  }

  // ------- instance data -------
  kwiver::vital::logger_handle_t m_logger;
  bool m_first;

  // MatLab wrapper parameters
  std::string m_matlab_program;       // name of matlab program
  vital::config_block_sptr m_config;

private:
  // MatLab support. The engine is allocated at the latest time.
  std::shared_ptr<matlab_engine> m_matlab_engine;

}; // end class matlab_image_filter::priv

// ==================================================================

matlab_image_filter::
matlab_image_filter()
  : d( new priv )
{
  attach_logger( "arrows.matlab.matlab_image_filter" );
  d->m_logger = logger();
}

 matlab_image_filter::
~matlab_image_filter()
{ }

// ------------------------------------------------------------------
vital::config_block_sptr
matlab_image_filter::
get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  config->set_value( "program_file", d->m_matlab_program,
                     "File name of the matlab image object filter program to run." );

  return config;
}

// ------------------------------------------------------------------
void
matlab_image_filter::
set_configuration(vital::config_block_sptr config)
{
  d->m_config = config;

  // Load specified program file into matlab engine
  d->m_matlab_program = config->get_value<std::string>( "program_file" );
}

// ------------------------------------------------------------------
bool
matlab_image_filter::
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
kwiver::vital::image_container_sptr
matlab_image_filter::
filter( kwiver::vital::image_container_sptr image_data)
{
  d->initialize_once();

  // convert image container to matlab image
  MxArraySptr mx_image = convert_mx_image( image_data );

  d->engine()->put_variable( "in_image", mx_image );
  d->eval( "global out_image; out_image=apply_filter(in_image)" );

  MxArraySptr out_image = d->engine()-> get_variable( "out_image" ); // throws
  d->check_result();

  kwiver::vital::image_container_sptr kv_image = convert_mx_image( out_image );

  return kv_image;
}

} } } // end namespace
