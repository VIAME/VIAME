/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * \brief Implementation of detected object set csv output
 */

#include "detected_object_set_output_csv.h"

#include <time.h>


namespace kwiver {
namespace arrows {
namespace core {

// ------------------------------------------------------------------
class detected_object_set_output_csv::priv
{
public:
  priv( detected_object_set_output_csv* parent)
    : m_parent( parent )
    , m_logger( kwiver::vital::get_logger( "detected_object_set_output_csv" ) )
    , m_first( true )
    , m_frame_number( 1 )
    , m_delim( "," )
  { }

  ~priv() { }

  detected_object_set_output_csv* m_parent;
  kwiver::vital::logger_handle_t m_logger;
  bool m_first;
  int m_frame_number;
  std::string m_delim;
};


// ==================================================================
detected_object_set_output_csv::
detected_object_set_output_csv()
  : d( new detected_object_set_output_csv::priv( this ) )
{
}


detected_object_set_output_csv::
~detected_object_set_output_csv()
{
}


// ------------------------------------------------------------------
void
detected_object_set_output_csv::
set_configuration(vital::config_block_sptr config)
{
  d->m_delim = config->get_value<std::string>( "delimiter", d->m_delim );
}


// ------------------------------------------------------------------
bool
detected_object_set_output_csv::
check_configuration(vital::config_block_sptr config) const
{
  return true;
}


// ------------------------------------------------------------------
void
detected_object_set_output_csv::
write_set( const kwiver::vital::detected_object_set_sptr set, std::string const& image_name )
{

  if (d->m_first)
  {
    std::time_t rawtime;
    struct tm * timeinfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    char* cp =  asctime( timeinfo );
    cp[ strlen( cp )-1 ] = 0; // remove trailing newline
    const std::string atime( cp );

    // Write file header(s)
    stream() << "# 1: image-index" << d->m_delim
             << "2:file-name" << d->m_delim
             << "3:TL-x" << d->m_delim
             << "4:TL-y" << d->m_delim
             << "5:BR-x" << d->m_delim
             << "6:BR-y" << d->m_delim
             << "7:confidence" << d->m_delim
             <<"{class-name" << d->m_delim << "score}" << d->m_delim << "..."
             << std::endl

      // Provide some provenience to the file. Could have a config
      // parameter that is copied to the file as a configurable
      // comment or marker.
             << "# Written on: " << atime
             << "   by: detected_object_set_output_csv"
             << std::endl;

    d->m_first = false;
  } // end first

  // process all detections
  auto ie =  set->cend();
  for ( auto det = set->cbegin(); det != ie; ++det )
  {
    const kwiver::vital::bounding_box_d bbox( (*det)->bounding_box() );
    stream() << d->m_frame_number << d->m_delim
             << image_name << d->m_delim
             << bbox.min_x() << d->m_delim // 2: TL-x
             << bbox.min_y() << d->m_delim // 3: TL-y
             << bbox.max_x() << d->m_delim // 4: BR-x
             << bbox.max_y() << d->m_delim // 5: BR-y
             << (*det)->confidence()          // 6: confidence value
      ;

    // Process classifications if there are any
    const auto dot( (*det)->type() );
    if ( dot )
    {
      const auto name_list( dot->class_names() );
      for( auto name : name_list )
      {
        // Write out the <name> <score> pair
        stream() << d->m_delim << name << d->m_delim << dot->score( name );
      } // end foreach
    }

    stream() << std::endl;

  } // end foreach

  // Put each set on a new frame
  ++d->m_frame_number;
}

} } } // end namespace
