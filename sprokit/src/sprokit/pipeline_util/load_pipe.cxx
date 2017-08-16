/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
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
 * \file load_pipe.cxx
 *
 * \brief Implementation of the pipeline declaration loading.
 */

#include "load_pipe.h"
#include "load_pipe_exception.h"

#if defined(_WIN32) || defined(_WIN64)
#include <sprokit/pipeline_util/include-paths.h>
#endif

#include "pipe_parser.h"

#include <vital/vital_foreach.h>
#include <vital/vital_types.h>

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline_util/pipe_parser.h>

#include <kwiversys/SystemTools.hxx>

#include <fstream>
#include <istream>
#include <sstream>
#include <string>

namespace sprokit {

static std::string const default_include_dirs = std::string( DEFAULT_PIPE_INCLUDE_PATHS );
static std::string const sprokit_include_envvar = std::string( "SPROKIT_PIPE_INCLUDE_PATH" );

// ------------------------------------------------------------------
pipe_blocks
load_pipe_blocks_from_file( kwiver::vital::path_t const& fname )
{
  sprokit::pipe_parser the_parser;

  kwiver::vital::path_list_t path_list;
  kwiversys::SystemTools::GetPath( path_list, sprokit_include_envvar.c_str() );
  if ( ! path_list.empty() )
  {
    the_parser.add_search_path( path_list );
  }

  std::ifstream input( fname );
  if ( ! input )
  {
    throw sprokit::file_no_exist_exception( fname );
  }

  return the_parser.parse_pipeline( input, fname );
}


// ------------------------------------------------------------------
pipe_blocks
load_pipe_blocks( std::istream& istr, std::string const& def_file )
{
  sprokit::pipe_parser the_parser;

  kwiver::vital::path_list_t path_list;
  kwiversys::SystemTools::GetPath( path_list, sprokit_include_envvar.c_str() );
  if ( ! path_list.empty() )
  {
    the_parser.add_search_path( path_list );
  }

  return the_parser.parse_pipeline( istr, def_file );
}


// ------------------------------------------------------------------
cluster_blocks
load_cluster_blocks_from_file( kwiver::vital::path_t const& fname )
{
  sprokit::pipe_parser the_parser;

  kwiver::vital::path_list_t path_list;
  kwiversys::SystemTools::GetPath( path_list, sprokit_include_envvar.c_str() );
  if ( ! path_list.empty() )
  {
    the_parser.add_search_path( path_list );
  }

  std::ifstream input( fname );
  if ( ! input )
  {
    throw sprokit::file_no_exist_exception( fname );
  }

  return the_parser.parse_cluster( input, fname );
}


// ------------------------------------------------------------------
cluster_blocks
load_cluster_blocks( std::istream& istr, std::string const& def_file )
{
  sprokit::pipe_parser the_parser;

  kwiver::vital::path_list_t path_list;
  kwiversys::SystemTools::GetPath( path_list, sprokit_include_envvar.c_str() );
  if ( ! path_list.empty() )
  {
    the_parser.add_search_path( path_list );
  }

  return the_parser.parse_cluster( istr, def_file );
}

} // end namespace
