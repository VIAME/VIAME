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

#include "write_track_descriptor_set_csv.h"

#include <time.h>

#include <vital/vital_foreach.h>

namespace kwiver {
namespace arrows {
namespace core {

// -------------------------------------------------------------------------------
class write_track_descriptor_set_csv::priv
{
public:
  priv( write_track_descriptor_set_csv* parent)
    : m_parent( parent )
    , m_logger( kwiver::vital::get_logger( "write_track_descriptor_set_csv" ) )
    , m_first( true )
    , m_frame_number( 1 )
    , m_delim( "," )
  { }

  ~priv() { }

  write_track_descriptor_set_csv* m_parent;
  kwiver::vital::logger_handle_t m_logger;
  bool m_first;
  int m_frame_number;
  std::string m_delim;
};


// ===============================================================================
write_track_descriptor_set_csv
::write_track_descriptor_set_csv()
  : d( new write_track_descriptor_set_csv::priv( this ) )
{
}


write_track_descriptor_set_csv
::~write_track_descriptor_set_csv()
{
}


// -------------------------------------------------------------------------------
void
write_track_descriptor_set_csv
::set_configuration(vital::config_block_sptr config)
{
  d->m_delim = config->get_value<std::string>( "delimiter", d->m_delim );
}


// -------------------------------------------------------------------------------
bool
write_track_descriptor_set_csv
::check_configuration(vital::config_block_sptr config) const
{
  return true;
}


// -------------------------------------------------------------------------------
void
write_track_descriptor_set_csv
::write_set( const kwiver::vital::track_descriptor_set_sptr set )
{

  if( d->m_first )
  {
    // Write file header(s)
    stream() << "# 1:descriptor_id, "
             << "2:descriptor_type, "
             << "3:track_references, "
             << "4:descriptor_data_vector, "
             << "5:history_vector"
             << std::endl;

    d->m_first = false;
  } // end first

  // Get detections from set
  VITAL_FOREACH( auto desc, *set )
  {
    if( !desc )
    {
      continue;
    }

    stream() << "" << d->m_delim;
    stream() << desc->get_type() << d->m_delim;
    stream() << "" << d->m_delim;
    stream() << desc->get_descriptor() << d->m_delim;

    // Process classifications if there are any
    VITAL_FOREACH( auto hist, desc->get_history() )
    {
      // TODO
    }

    stream() << std::endl;
  }

  // Put each set on a new frame
  ++d->m_frame_number;
}

} } } // end namespace
