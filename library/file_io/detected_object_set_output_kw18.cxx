// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "detected_object_set_output_kw18.h"

#include <viame/algorithm_framework/util/tokenize.h>
#include <viame/algorithm_framework/vital_config.h>

#include <ctime>
#include <fstream>
#include <memory>
#include <vector>

#if ( __GNUC__ == 4 && __GNUC_MINOR__ < 5 && !defined( __clang__ ) )
#include <cstdatomic>
#else
#include <atomic>
#endif

namespace kwiver {

namespace arrows {

namespace core {

/// This format should only be used for tracks.
///
/// \li Column(s) 1: Track-id
/// \li Column(s) 2: Track-length (# of detections)
/// \li Column(s) 3: Frame-number (-1 if not available)
/// \li Column(s) 4-5: Tracking-plane-loc(x,y) (Could be same as World-loc)
/// \li Column(s) 6-7: Velocity(x,y)
/// \li Column(s) 8-9: Image-loc(x,y)
/// \li Column(s) 10-13: Img-bbox(TL_x,TL_y,BR_x,BR_y) (location of top-left &
/// bottom-right vertices)
/// \li Column(s) 14: Area (0 - when not available)
/// \li Column(s) 15-17: World-loc(x,y,z) (longitude, latitude, 0 - when not
/// available)
/// \li Column(s) 18: Timesetamp(-1 if not available)
/// \li Column(s) 19: Track-confidence(-1_when_not_available)

// ----------------------------------------------------------------------------
class detected_object_set_output_kw18::priv
{
public:
  priv( detected_object_set_output_kw18& parent )
    : parent( parent ),
      m_first( true ),
      m_frame_number( 1 )
  {}

  ~priv() {}

  void read_all();

  detected_object_set_output_kw18& parent;

  // Configuration values
  bool c_write_tot() { return parent.c_write_tot; }
  std::string c_tot_field1_ids() { return parent.c_tot_field1_ids; }
  std::string c_tot_field2_ids() { return parent.c_tot_field2_ids; }

  bool m_first;
  int m_frame_number;

  std::unique_ptr< std::ofstream > m_tot_writer;

  // we need to recalculate these values each time because between invocations
  // of set/get configuration parent.set_path() will make these out of sync.
  std::vector< std::string >
  m_parsed_tot_ids1()
  {
    auto const& field1_ids = parent.c_tot_field1_ids;
    std::vector< std::string > result;
    kwiver::vital::tokenize(
      field1_ids, result, ",;",
      kwiver::vital::TokenizeTrimEmpty  );

    return result;
  }

  std::vector< std::string >
  m_parsed_tot_ids2()
  {
    auto const& field2_ids = parent.c_tot_field2_ids;
    std::vector< std::string > result;
    kwiver::vital::tokenize(
      field2_ids, result, ",;",
      kwiver::vital::TokenizeTrimEmpty  );

    return result;
  }
};

// ----------------------------------------------------------------------------
void
detected_object_set_output_kw18
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "arrows.core.detected_object_set_output_kw18" );
}

detected_object_set_output_kw18::
~detected_object_set_output_kw18()
{
  if( d->m_tot_writer && d->m_tot_writer->is_open() )
  {
    d->m_tot_writer->close();
  }
}

// ----------------------------------------------------------------------------
bool
detected_object_set_output_kw18
::check_configuration( [[maybe_unused]] vital::config_block_sptr config ) const
{
  if( d->c_write_tot() && d->c_tot_field1_ids().empty() )
  {
    return false;
  }

  if( d->c_write_tot() && d->c_tot_field2_ids().empty() )
  {
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------------
void
detected_object_set_output_kw18
::write_set(
  const kwiver::vital::detected_object_set_sptr set,
  [[maybe_unused]] std::string const& image_name )
{
  if( d->m_first )
  {
    std::time_t rawtime;
    struct tm* timeinfo;

    time( &rawtime );
    timeinfo = localtime( &rawtime );

    char* cp = asctime( timeinfo );
    cp[ strlen( cp ) - 1 ] = 0; // remove trailing newline

    const std::string atime( cp );

    // Write file header(s)
    stream() << "# 1:Track-id "
             << "2:Track-length "
             << "3:Frame-number "
             << "4:Tracking-plane-loc(x) "
             << "5:Tracking-plane-loc(y) "
             << "6:velocity(x) "
             << "7:velocity(y) "

             << "8:Image-loc(x)"
             << " 9:Image-loc(y)"
             << " 10:Img-bbox(TL_x)"
             << " 11:Img-bbox(TL_y)"
             << " 12:Img-bbox(BR_x)"
             << " 13:Img-bbox(BR_y)"
             << " 14:Area"

             << " 15:World-loc(x)"
             << " 16:World-loc(y)"
             << " 17:World-loc(z)"
             << " 18:timestamp"
             << " 19:track-confidence"
             << std::endl

      // Provide some provenience to the file. Could have a config
      // parameter that is copied to the file as a configurable
      // comment or marker.

             << "# Written on: " << atime
             << "   by: detected_object_set_output_kw18"
             << std::endl;

    d->m_first = false;

    if( d->c_write_tot() )
    {
      std::size_t ext_ind = filename().find_last_of( "." );
      std::string tot_fn = filename().substr( 0, ext_ind ) + ".txt";

      d->m_tot_writer.reset( new std::ofstream( tot_fn ) );
    }
  } // end first

  // process all detections
  auto ie = set->cend();
  for( auto det = set->cbegin(); det != ie; ++det )
  {
    const kwiver::vital::bounding_box_d bbox( ( *det )->bounding_box() );
    double ilx = ( bbox.min_x() + bbox.max_x() ) / 2.0;
    double ily = ( bbox.min_y() + bbox.max_y() ) / 2.0;

    static std::atomic< size_t > id_counter( 0 );
    auto const l_id = id_counter++;

    stream() << l_id                  // 1: track id
             << " 1 "               // 2: track length
             << d->m_frame_number - 1 // 3: frame number / set number
             << " 0 "               // 4: tracking plane x
             << " 0 "               // 5: tracking plane y
             << "0 "                // 6: velocity x
             << "0 "                // 7: velocity y
             << ilx << " "          // 8: image location x
             << ily << " "          // 9: image location y
             << bbox.min_x() << " " // 10: TL-x
             << bbox.min_y() << " " // 11: TL-y
             << bbox.max_x() << " " // 12: BR-x
             << bbox.max_y() << " " // 13: BR-y
             << bbox.area() << " "  // 14: area
             << "0 "                // 15: world-loc x
             << "0 "                // 16: world-loc y
             << "0 "                // 17: world-loc z
             << "-1 "                // 18: timestamp
             << ( *det )->confidence() // 19: confidence
             << std::endl;

    // optionally write tot to corresponding file
    if( d->c_write_tot() )
    {
      vital::detected_object_type_sptr clf = ( *det )->type();

      double f1 = 0.0, f2 = 0.0, f3 = 0.0;

      for( auto const& id : d->m_parsed_tot_ids1() )
      {
        if( clf->has_class_name( id ) )
        {
          f1 = std::max( f1, clf->score( id ) );
        }
      }

      for( auto const& id : d->m_parsed_tot_ids2() )
      {
        if( clf->has_class_name( id ) )
        {
          f2 = std::max( f2, clf->score( id ) );
        }
      }

      f3 = 1.0 - f2 - f1;

      ( *d->m_tot_writer ) << l_id << " " << f1 << " " << f2 << " " << f3 <<
        std::endl;
    } // end write_tot
  } // end foreach

  // Put each set on a new frame
  ++d->m_frame_number;
}

} // namespace core

} // namespace arrows

}     // end namespace
