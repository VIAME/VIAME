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
 * @file
 * @brief The track_oracle file format implementation for NOAA CSV files.
 *
 */

#include "file_format_noaa_csv.h"

#include <fstream>

#include <track_oracle/utils/tokenizers.h>
#include <vital/util/string.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::string;
using std::getline;
using std::ifstream;
using std::istream;
using std::vector;

namespace // anon
{

bool
read_non_comment_csv_fields( istream& is, vector< string >& tokens )
{
  while (kwiver::track_oracle::csv_tokenizer::get_record( is, tokens ))
  {
    if (! tokens.empty() )
    {
      string& first_token = tokens[0];
      kwiver::vital::left_trim( first_token );
      if (first_token[0] != '#')
      {
        return true;
      }
      else
      {
        // keep going
      }
    }
    else
    {
      return false;
    }
  }
  return false;
}

} // ...anon


namespace kwiver {
namespace track_oracle {

file_format_noaa_csv
::file_format_noaa_csv()
  : file_format_base( TF_NOAA_CSV, "NOAA CSV" )
{
  this->globs.push_back("*.noaa.csv");
}


file_format_noaa_csv
::~file_format_noaa_csv()
{
}

bool
file_format_noaa_csv
::inspect_file( const string& fn ) const
{
  //
  // For now, look for a non-comment line; return true if >= 10 fields
  //
  ifstream is(fn.c_str());
  if ( ! is ) return false;

  vector< string > tokens;
  return ( read_non_comment_csv_fields( is, tokens )
           &&
           (tokens.size() >= 10) );
}

bool
file_format_noaa_csv
::read(const string& fn, track_handle_list_type& tracks) const
{
  ifstream ifs(fn.c_str());
  return ifs && this->read(ifs, tracks);
}

bool
file_format_noaa_csv
::read( istream& is, track_handle_list_type& tracks) const
{
  //
  // Each line is, for now, a single-track detection
  //

  track_noaa_csv_type trk;
  vector< string > tokens;
  bool okay = true;

  while (okay && is)
  {
    if ( ! read_non_comment_csv_fields( is, tokens ))
    {
      continue;
    }

    auto ntokens = tokens.size();
    if (ntokens < 10)
    {
      okay = false;
      continue;
    }

    try
    {
      track_handle_type t = trk.create();
      trk( t ).det_id() = stoi( tokens[0] );
      frame_handle_type f = trk.create_frame();
      trk[ f ].frame_number() = stoi( tokens[2] );
      trk[ f ].bounding_box() =
        vgl_box_2d<double>(
          vgl_point_2d<double>( stod( tokens[3] ), stod( tokens[4] )),
          vgl_point_2d<double>( stod( tokens[5] ), stod( tokens[6] )));
      trk( t ).relevancy() = stod( tokens[7] );


      kpf_cset_type cset;
      for (auto i=9; okay && (i<ntokens); i+=2)
      {
        if (i+1 >= ntokens)
        {
          LOG_ERROR( main_logger, "NOAA CSV: line with " << ntokens << " tokens; can't parse into species / confidence pairs" );
          okay = false;
          continue;
        }

        const string& species = tokens[i];
        double confidence = stod( tokens[i+1] );
        cset.insert( make_pair( species, confidence ));
      }

      trk[ f ].species_cset() = cset;

      tracks.push_back( t );
    }
    catch (const std::invalid_argument& e)
    {
      LOG_ERROR( main_logger, "NOAA CSV: error converting argument: " << e.what() );
      okay = false;
    }

  } // ... while okay

  return okay;
}



} // ...track_oracle
} // ...kwiver
