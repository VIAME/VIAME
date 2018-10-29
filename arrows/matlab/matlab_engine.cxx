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
