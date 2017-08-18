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
 * \brief Implementation of save wrapping functionality.
 */

#include "write_object_track_set.h"

#include <vital/algo/algorithm.txx>
#include <vital/exceptions/io.h>
#include <vital/vital_types.h>

#include <kwiversys/SystemTools.hxx>

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::write_object_track_set);
/// \endcond


namespace kwiver {
namespace vital {
namespace algo {

write_object_track_set
::write_object_track_set()
  : m_stream( 0 )
  , m_stream_owned( false )
{
  attach_logger( "write_object_track_set" );
}


write_object_track_set
::~write_object_track_set()
{
}


// ------------------------------------------------------------------------------------
void
write_object_track_set
::open( std::string const& filename )
{
  // try to open the file
  std::ostream* file( new std::ofstream( filename ) );

  if( ! file )
  {
    kwiver::vital::file_not_found_exception( filename, "open failed"  );
  }

  m_stream = file;
  m_stream_owned = true;
  m_filename = filename;
}


// ------------------------------------------------------------------------------------
void
write_object_track_set
::use_stream( std::ostream* strm )
{
  m_stream = strm;
  m_stream_owned = false;
}


// ------------------------------------------------------------------------------------
void
write_object_track_set
::close()
{
  if( m_stream_owned )
  {
    delete m_stream;
  }

  m_stream = 0;
}


// ------------------------------------------------------------------------------------
std::ostream&
write_object_track_set
::stream()
{
  return *m_stream;
}


// ------------------------------------------------------------------------------------
std::string const&
write_object_track_set
::filename()
{
  return m_filename;
}


} } } // end namespace
