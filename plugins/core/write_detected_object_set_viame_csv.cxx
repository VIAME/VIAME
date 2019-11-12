/*ckwg +29
 * Copyright 2018-2021 by Kitware, Inc.
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

#include "write_detected_object_set_viame_csv.h"

#include "notes_to_attributes.h"

#include <vital/util/tokenize.h>

#include <memory>
#include <vector>
#include <fstream>
#include <ctime>

#if ( __GNUC__ == 4 && __GNUC_MINOR__ < 5 && !defined(__clang__) )
  #include <cstdatomic>
#else
  #include <atomic>
#endif


namespace viame {


// --------------------------------------------------------------------------------
class write_detected_object_set_viame_csv::priv
{
public:
  priv( write_detected_object_set_viame_csv* parent)
    : m_parent( parent )
    , m_first( true )
    , m_frame_number( 0 )
    , m_write_frame_number( true )
    , m_stream_identifier( "" )
    , m_model_identifier( "" )
    , m_version_identifier( "" )
  {}

  ~priv() {}

  write_detected_object_set_viame_csv* m_parent;
  bool m_first;
  int m_frame_number;
  bool m_write_frame_number;
  std::string m_stream_identifier;
  std::string m_model_identifier;
  std::string m_version_identifier;
};


// ================================================================================
write_detected_object_set_viame_csv
::write_detected_object_set_viame_csv()
  : d( new write_detected_object_set_viame_csv::priv( this ) )
{
  attach_logger( "viame.core.write_detected_object_set_viame_csv" );
}


write_detected_object_set_viame_csv
::~write_detected_object_set_viame_csv()
{
}


// --------------------------------------------------------------------------------
void
write_detected_object_set_viame_csv
::set_configuration( kwiver::vital::config_block_sptr config_in )
{
  kwiver::vital::config_block_sptr config = this->get_configuration();

  d->m_write_frame_number =
    config->get_value< bool >( "write_frame_number", d->m_write_frame_number );
  d->m_stream_identifier =
    config->get_value< std::string >( "stream_identifier", d->m_stream_identifier );
  d->m_model_identifier =
    config->get_value< std::string >( "model_identifier", d->m_model_identifier );
  d->m_version_identifier =
    config->get_value< std::string >( "version_identifier", d->m_version_identifier );

  config->merge_config( config_in );
}


// --------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
write_detected_object_set_viame_csv
::get_configuration() const
{
  // get base config from base class
  kwiver::vital::config_block_sptr config = algorithm::get_configuration();

  // Class parameters
  config->set_value( "write_frame_number", d->m_write_frame_number,
    "Write a frame number for the unique frame ID field (as opposed to a string "
    "identifier) for column 3 in the output csv." );
  config->set_value( "stream_identifier", d->m_stream_identifier,
    "Optional fixed video name over-ride to write to output column 2 in the csv." );
  config->set_value( "model_identifier", d->m_model_identifier,
    "Model identifier string to write to the header or the csv." );
  config->set_value( "version_identifier", d->m_version_identifier,
    "Version identifier string to write to the header or the csv." );

  return config;
}


// --------------------------------------------------------------------------------
bool
write_detected_object_set_viame_csv
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// --------------------------------------------------------------------------------
void
write_detected_object_set_viame_csv
::write_set( const kwiver::vital::detected_object_set_sptr set,
           std::string const& image_name )
{
  if( d->m_first )
  {
    std::time_t rawtime;
    struct tm * timeinfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    char* cp =  asctime( timeinfo );
    cp[ strlen( cp )-1 ] = 0; // remove trailing newline
    const std::string atime( cp );

    // Write file header(s)
    stream() << "# 1: Detection or Track-id,"
             << "  2: Video or Image Identifier,"
             << "  3: Unique Frame Identifier,"
             << "  4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y),"
             << "  8: Detection or Length Confidence,"
             << "  9: Target Length (0 or -1 if invalid),"
             << "  10-11+: Repeated Species, Confidence Pairs or Attributes"
             << std::endl;

    stream() << "# Written on: " << atime
             << "   by: write_detected_object_set_viame_csv"
             << std::endl;

    if( !d->m_model_identifier.empty() )
    {
      stream() << "# Computed using model identifier: "
               << d->m_model_identifier
               << std::endl;
    }

    if( !d->m_version_identifier.empty() )
    {
      stream() << "# Computed using software version: "
               << d->m_version_identifier
               << std::endl;
    }

    d->m_first = false;
  } // end first

  // skip frames with no image name
  if( image_name.empty() )
  {
    return;
  }

  // process all detections if a valid input set was provided
  if( !set )
  {
    ++d->m_frame_number;
    return;
  }

  auto ie = set->cend();

  for( auto det = set->cbegin(); det != ie; ++det )
  {
    const kwiver::vital::bounding_box_d bbox( (*det)->bounding_box() );

    static std::atomic<unsigned> id_counter( 0 );
    const unsigned det_id = id_counter++;

    const std::string video_id = ( !d->m_stream_identifier.empty() ?
                                    d->m_stream_identifier :
                                    image_name );

    stream() << det_id << ","               // 1: track id
             << video_id << ",";            // 2: video or image id

    if( d->m_write_frame_number )
    {
      stream() << d->m_frame_number << ","; // 3: frame number
    }
    else
    {
      stream() << image_name << ",";        // 3: frame identfier
    }

    stream() << bbox.min_x() << ","         // 4: TL-x
             << bbox.min_y() << ","         // 5: TL-y
             << bbox.max_x() << ","         // 6: BR-x
             << bbox.max_y() << ","         // 7: BR-y
             << (*det)->confidence() << "," // 8: confidence
             << "0";                        // 9: length

    const auto dot = (*det)->type();

    if( dot )
    {
      const auto name_list( dot->class_names() );

      for( auto name : name_list )
      {
        // Write out the <name> <score> pair
        stream() << "," << name << "," << dot->score( name );
      }
    }

    if( !(*det)->keypoints().empty() )
    {
      for( const auto& kp : (*det)->keypoints() )
      {
        stream() << "," << "(kp) " << kp.first;
        stream() << " " << kp.second.value()[0] << " " << kp.second.value()[1];
      }
    }

    if( !(*det)->notes().empty() )
    {
      stream() << notes_to_attributes( (*det)->notes(), "," );
    }

    stream() << std::endl;
  }

  // Flush stream to prevent buffer issues
  stream().flush();

  // Put each set on a new frame
  ++d->m_frame_number;
}

} // end namespace
