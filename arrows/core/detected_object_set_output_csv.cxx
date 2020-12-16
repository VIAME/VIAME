// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of detected object set csv output
 */

#include "detected_object_set_output_csv.h"

#include <ctime>

namespace kwiver {
namespace arrows {
namespace core {

// ------------------------------------------------------------------
class detected_object_set_output_csv::priv
{
public:
  priv( detected_object_set_output_csv* parent)
    : m_parent( parent )
    , m_first( true )
    , m_frame_number( 1 )
    , m_delim( "," )
  { }

  ~priv() { }

  detected_object_set_output_csv* m_parent;
  bool m_first;
  int m_frame_number;
  std::string m_delim;
};

// ==================================================================
detected_object_set_output_csv::
detected_object_set_output_csv()
  : d( new detected_object_set_output_csv::priv( this ) )
{
  attach_logger( "arrows.core.detected_object_set_output_csv" );
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
check_configuration( VITAL_UNUSED vital::config_block_sptr config) const
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
    const auto cm( (*det)->type() );
    if ( cm )
    {
      const auto name_list( cm->class_names() );
      for( auto name : name_list )
      {
        // Write out the <name> <score> pair
        stream() << d->m_delim << name << d->m_delim << cm->score( name );
      } // end foreach
    }

    stream() << std::endl;

  } // end foreach

  // Put each set on a new frame
  ++d->m_frame_number;
}

} } } // end namespace
