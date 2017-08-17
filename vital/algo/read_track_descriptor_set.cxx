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

/**
 * \file
 * \brief Implementation of load/save wrapping functionality.
 */

#include "read_track_descriptor_set.h"

#include <vital/algo/algorithm.txx>
#include <vital/exceptions/io.h>
#include <vital/vital_types.h>

#include <kwiversys/SystemTools.hxx>

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::read_track_descriptor_set);
/// \endcond


namespace kwiver {
namespace vital {
namespace algo {

read_track_descriptor_set
::read_track_descriptor_set()
  : m_stream( 0 )
  , m_stream_owned( false )
{
  attach_logger( "read_track_descriptor_set" );
}


read_track_descriptor_set
::~read_track_descriptor_set()
{
  if ( m_stream && m_stream_owned )
  {
    delete m_stream;
  }

  m_stream = 0;
}


// ------------------------------------------------------------------------------------
void
read_track_descriptor_set
::open( std::string const& filename )
{
    // Make sure that the given file path exists and is a file.
  if( ! kwiversys::SystemTools::FileExists( filename ) )
  {
    throw path_not_exists(filename);
  }

  if( kwiversys::SystemTools::FileIsDirectory( filename ) )
  {
    throw path_not_a_file(filename);
  }

  // try to open the file
  std::istream* file( new std::ifstream( filename ) );
  if( ! file )
  {
    kwiver::vital::file_not_found_exception( filename, "open failed"  );
  }

  m_stream = file;
  m_stream_owned = true;

  new_stream();
}


// ------------------------------------------------------------------------------------
void
read_track_descriptor_set
::use_stream( std::istream* strm )
{
  m_stream = strm;
  m_stream_owned = false;

  new_stream();
}


// ------------------------------------------------------------------------------------
void
read_track_descriptor_set
::close()
{
  if( m_stream_owned )
  {
    delete m_stream;
  }

  m_stream = 0;
}


// ------------------------------------------------------------------------------------
bool
read_track_descriptor_set
::at_eof() const
{
  if( m_stream )
  {
    return m_stream->eof();
  }
  else
  {
    return true; // really error
  }
}


// ------------------------------------------------------------------------------------
std::istream&
read_track_descriptor_set
::stream()
{
  return *m_stream;
}


// ------------------------------------------------------------------------------------
void
read_track_descriptor_set
::new_stream()
{
}

} } } // end namespace
