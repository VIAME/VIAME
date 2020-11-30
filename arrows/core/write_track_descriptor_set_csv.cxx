// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of track descriptor set csv output
 */

#include "write_track_descriptor_set_csv.h"

#include <vital/vital_config.h>

#include <time.h>

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
::check_configuration( VITAL_UNUSED vital::config_block_sptr config ) const
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
  for( auto desc : *set )
  {
    if( !desc )
    {
      continue;
    }

    stream() << desc->get_uid().value() << d->m_delim;
    stream() << desc->get_type() << d->m_delim;

    stream() << desc->get_track_ids().size() << d->m_delim;
    for( auto id : desc->get_track_ids() )
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
    for( auto h : desc->get_history() )
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
