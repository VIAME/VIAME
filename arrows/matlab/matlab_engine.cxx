// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for MatLab engine interface class.
 */

#include "matlab_engine.h"
#include "matlab_exception.h"

#include <sstream>

namespace kwiver {
namespace arrows {
namespace matlab {

// ------------------------------------------------------------------
matlab_engine::
matlab_engine()
  : m_logger( kwiver::vital::get_logger( "arrows.matlab.matlab_engine" ) )
  , m_engine_handle( 0 )
  , m_output_buffer( 0 )
{
  m_engine_handle = engOpen( "" );
  if ( 0 == m_engine_handle)
  {
    VITAL_THROW( matlab_exception, "Error opening MatLab engine" );
  }

  m_output_buffer = static_cast< char * >(malloc( 4096 ));
  int status = engOutputBuffer( m_engine_handle, m_output_buffer, 4096 );
  if ( status )
  {
    VITAL_THROW( matlab_exception,"Invalid engine handle in engOutputBuffer() call" );
  }
}

// ------------------------------------------------------------------
matlab_engine::
~matlab_engine()
{
  int status = engClose( m_engine_handle );
  m_engine_handle = 0;

  if ( status )
  {
    LOG_WARN( m_logger, "Error returned from closing MatLab engine: " << status );
  }

  free( m_output_buffer );
  m_output_buffer = 0;
}

// ------------------------------------------------------------------
void
matlab_engine::
eval( const std::string& cmd )
{
  int status = engEvalString( m_engine_handle, cmd.c_str() );
  if ( 1 == status )
  {
    VITAL_THROW( matlab_exception, "Engine session no longer active" );
  }
}

// ------------------------------------------------------------------
MxArraySptr
matlab_engine::
get_variable( const std::string& name )
{
  mxArray* var = engGetVariable( m_engine_handle, name.c_str() );
  if ( ! var )
  {
    std::stringstream str;
    str << "Variable \"" << name << "\" does not exist.";
    VITAL_THROW( matlab_exception, str.str() );
  }

  return MxArraySptr( new MxArray( var ) );
}

// ------------------------------------------------------------------
void
matlab_engine::
put_variable( const std::string& name, MxArraySptr val )
{
  int status = engPutVariable( m_engine_handle, name.c_str(), val.get()->get() );
  if ( status )
  {
    std::stringstream str;
    str << "Error assigning value to variable \"" << name << "\"";
    VITAL_THROW( matlab_exception, str.str() );
  }
}

// ------------------------------------------------------------------
bool
matlab_engine::
get_visible()
{
  bool retval( 0 );
  int status = engGetVisible( m_engine_handle, &retval );
  if ( status )
  {
    LOG_WARN( m_logger, "Error returned from engGetVisible()");
  }
  return retval;
}

// ------------------------------------------------------------------
void
matlab_engine::
set_visible( bool vis )
{
  int status = engSetVisible( m_engine_handle, vis );
  if ( status )
  {
    LOG_WARN( m_logger, "Error returned from engSetVisible()");
  }
}

// ------------------------------------------------------------------
std::string
matlab_engine::
output() const
{
  return std::string( m_output_buffer );
}

} } }     // end namesapce
