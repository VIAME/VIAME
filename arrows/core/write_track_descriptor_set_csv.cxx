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
 * \brief Implementation of track descriptor set csv output
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
    , m_write_raw_descriptor( true )
    , m_write_world_loc( false )
    , m_delim( "," )
    , m_sub_delim( " " )
  {}

  ~priv() {}

  write_track_descriptor_set_csv* m_parent;
  kwiver::vital::logger_handle_t m_logger;
  bool m_first;
  bool m_write_raw_descriptor;
  bool m_write_world_loc;
  std::string m_delim;
  std::string m_sub_delim;
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
::set_configuration( vital::config_block_sptr config )
{
  d->m_write_raw_descriptor =
    config->get_value<bool>( "write_raw_descriptor", d->m_write_raw_descriptor);
    config->get_value<bool>( "write_world_loc", d->m_write_world_loc);
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
    stream() << "# 1:descriptor_uid, "
             << "2:descriptor_type, "
             << "3:track_reference_size, "
             << "4:track_references, "
             << "5:descriptor_size, "
             << "6:descriptor_data_vector, "
             << "7:history_size, "
             << "8:history_vector"
             << std::endl;

    d->m_first = false;
  }

  // Get detections from set
  VITAL_FOREACH( auto desc, *set )
  {
    if( !desc )
    {
      continue;
    }

    stream() << desc->get_uid().value() << d->m_delim;
    stream() << desc->get_type() << d->m_delim;

    stream() << desc->get_track_ids().size() << d->m_delim;
    VITAL_FOREACH( auto id, desc->get_track_ids() )
    {
      stream() << id << d->m_sub_delim;
    }
    stream() << d->m_delim;

    if( d->m_write_raw_descriptor )
    {
      auto raw_sptr = desc->get_descriptor();
      stream() << raw_sptr->size() << d->m_delim;
      for( auto* value_ptr = raw_sptr->raw_data();
           value_ptr != raw_sptr->raw_data() + raw_sptr->size();
           value_ptr++ )
      {
        stream() << *value_ptr << d->m_sub_delim;
      }
      stream() << d->m_delim;
    }
    else
    {
      stream() << 0 << d->m_delim << " " << d->m_delim;
    }

    // Process classifications if there are any
    stream() << desc->get_history().size() << d->m_delim;
    VITAL_FOREACH( auto h, desc->get_history() )
    {
      stream() << h.get_timestamp().get_frame() << d->m_sub_delim;
      stream() << h.get_timestamp().get_time_usec() << d->m_sub_delim;

      stream() << h.get_image_location().min_x() << d->m_sub_delim;
      stream() << h.get_image_location().min_y() << d->m_sub_delim;
      stream() << h.get_image_location().max_x() << d->m_sub_delim;
      stream() << h.get_image_location().max_y() << d->m_sub_delim;

      if( d->m_write_world_loc )
      {
        stream() << h.get_world_location().min_x() << d->m_sub_delim;
        stream() << h.get_world_location().min_y() << d->m_sub_delim;
        stream() << h.get_world_location().max_x() << d->m_sub_delim;
        stream() << h.get_world_location().max_y() << d->m_sub_delim;
      }
    }
    stream() << std::endl;
  }
}

} } } // end namespace
