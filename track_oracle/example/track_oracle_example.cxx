/*ckwg +5
 * Copyright 2010-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <stdexcept>
#include <fstream>
#include <ostream>

//
// This is a short example demonstrating how to use the track oracle.
//

// Search for the string "Hmm" to see current "gotchas" in the design.

// step 1: include the two fundamental types, track_base and track_field.
//
// track_base is the base class for your track; create your track by
// inheriting from track_base (see below.)  track_base provides two
// public data members: 'Track' and 'Frame'.  You define your track
// by associating track_fields (next) with Track and Frame.
//
// track_field is used to establish an association between three concepts
// which define an element of a track:
//
// 1) the data type (int, double, vgl_box<double>, etc. )
// 2) the universal cross-project name of that data type
// 3) whether or not the element is associated with the track as a whole
//    or with frames on the track.
//


#include <vital/vital_config.h>
#include <track_oracle/core/track_oracle_export.h>
#include <track_oracle/core/element_descriptor.h>
#include <track_oracle/core/track_base.h>
#include <track_oracle/core/track_field.h>
#include <track_oracle/core/track_oracle_core.h>
#include <track_oracle/core/schema_algorithm.h>
#include <track_oracle/data_terms/data_terms.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::pair;
using std::runtime_error;
using std::string;
using std::vector;

// Each unique instance of a track_field (a track as a whole or a frame
// of a track) is referenced by its opaque oracle_entry_handle_type.

using ::kwiver::track_oracle::track_oracle_core;
using ::kwiver::track_oracle::track_field;
using ::kwiver::track_oracle::oracle_entry_handle_type;
using ::kwiver::track_oracle::track_handle_type;
using ::kwiver::track_oracle::track_handle_list_type;
using ::kwiver::track_oracle::frame_handle_type;
using ::kwiver::track_oracle::track_oracle_frame_view;
using ::kwiver::track_oracle::frame_handle_list_type;
using ::kwiver::track_oracle::element_descriptor;
using ::kwiver::track_oracle::field_handle_type;

// Define your track elements as *references* to track_fields.  Associate
// these references with either the Track or Frame members from the track_base
// in the constructor.  Use the Curiously Recurring Template Pattern for
// static polymorphism (in particular, to provide the operator() for frame
// access.)  (Thanks to Amitha for suggesting this!)

struct my_particular_track: public ::kwiver::track_oracle::track_base< my_particular_track >
{
  // track level data
  track_field<double>& score;
  track_field<string>& label;

  // frame level data
  track_field<unsigned long long>& timestamp;

  my_particular_track():
    score( Track.add_field<double>( "score") ),
    label( Track.add_field<string>( "label" ) ),
    timestamp( Frame.add_field<unsigned long long>( "timestamp_usecs" ) )
  {
  }
};


// The resulting structure, in this case 'my_particular_track', is not
// a structure you instantiate multiple times to get multiple instances of
// a track.  Instead, it acts as a type-safe "window" into the (invisible-to-
// the-user) "cloud of tracks"; see below.

// We can define a separate track type which shares a field with
// my_particular_track.  Because my_particular_track.label and
// a_differerent_track.activity_label were both initialized with the same
// data name (e.g. "label"), they both refer to the same data store in
// the universal "cloud of tracks."

struct a_different_track: public ::kwiver::track_oracle::track_base< a_different_track >
{
  // this track only uses labels.  Note that the code-level name of
  // the data member is irrelevant...
  track_field<string>& activity_label;

  // ...what matters is the name passed here!
  a_different_track():
    activity_label( Track.add_field<string>( "label" ))
  {}
};

// (Based on a suggestion from Matthew) You can also inherit from another
// track.  Adding a field twice throws an exception when you instantiate
// the schema.

struct derived_track_throws_exception: public ::kwiver::track_oracle::track_base< derived_track_throws_exception, my_particular_track >
{
  track_field<double>& score;

  derived_track_throws_exception():
    score( Track.add_field< double >( "score" ))
  {}
};

struct derived_track: public ::kwiver::track_oracle::track_base< derived_track, my_particular_track >
{
  track_field<double>& foo;

  derived_track():
    foo( Track.add_field< double >( "relevancy" ))
  {}
};


int main( int argc, char *argv[] )
{

  // As shown below, tracks exist independently of instances of the
  // track data structure; each instance of a track is associated
  // with a handle.
  track_handle_type track_handle;

  {
    // instantiate your track...
    my_particular_track t;

    LOG_INFO( main_logger, "Output uncreated track_base:");
    LOG_INFO( main_logger, t);
    LOG_INFO( main_logger, "\ndone");

    // you now have the definition of your track, but no actual tracks.
    // To create an instance of your track, call create():
    track_handle = t.create();

    // The instance of your track is now pointing to the track whose opaque
    // handle is stored in track_handle.  All accesses to the fields of your
    // track go through your instance:

    t.score() = 5.0;
    t.label() = "walking";

    LOG_INFO( main_logger, "Output after two values set:");
    LOG_INFO( main_logger, t);
    LOG_INFO( main_logger, "\ndone");

    // The fields do not exist until you set them.  If you read them before
    // setting them, the default value for that type is returned.  (TBD: define
    // must-exist semantics.)

    // Note that [] is used to access frames.

    // Frames are created for the track in the same fashion:
    for (unsigned i=0; i<5; ++i)
    {
      frame_handle_type frame_handle = t.create_frame();
      t[frame_handle].timestamp() = 100+i;
    }

    // If all the data types you've used have operator<<() defined,
    // then you get the operator<<() for the track for free.  (If
    // not, then your program won't compile.)
    LOG_INFO( main_logger, t << "");

    // We can get a list of the frames...
    ::kwiver::track_oracle::frame_handle_list_type frames = track_oracle_core::get_frames( track_handle );
    frame_handle_type first_frame = frames[0];
    frame_handle_type last_frame = frames[ frames.size()-1 ];

    // and access different frames via the track's operator[]:
    t[first_frame].timestamp() = 1000;
    t[last_frame].timestamp() = 2000;

    //
    // You can dynamically create fields; run as $0 timestamp to see
    // frame 3's timestamp, or as $0 foobar to see that frame 3
    // has no foobar field
    if (argc > 1)
    {
      track_field<int> probe(argv[1]);
      frame_handle_type f3 = frames[3];
      if (probe.exists( f3.row ))
      {
        LOG_INFO( main_logger, "Frame 3's '" << argv[1] << "' value is " << probe(f3.row) << "");
      }
      else
      {
        LOG_INFO( main_logger, "Frame 3 has no '" << argv[1] << "' field");
      }
    }

    LOG_INFO( main_logger, "After changing first and last timestamps");
    LOG_INFO( main_logger, t << "");

    LOG_INFO( main_logger, "About to write");
    // write it out to be sure
    track_handle_list_type tracks;
    tracks.push_back( track_handle );
    std::ofstream os ("./foo.kwiver");
    track_oracle_core::write_kwiver( os, tracks );
    LOG_INFO( main_logger, "wrote");

    // delete some frames
    t.remove_frame( first_frame );
    LOG_INFO( main_logger, "After deleting first frame\n" << t << "");
    t.remove_frame( last_frame );
    LOG_INFO( main_logger, "After deleting last frame\n" << t << "");


    // Now we go out of scope, taking the instance of our
    // track data structure along with it.
  }

  LOG_INFO( main_logger, "Out of scope");

  // What happened to our track data?
  //
  // It's still there:
  //
  // THE TRACK'S DATA IS NOT ASSOCIATED WITH INSTANCES OF OUR TRACK DATA STRUCTURE.
  //
  // THE TRACK'S DATA IS ACCESSED VIA INSTANCES OF OUR TRACK DATA STRUCTURE.
  //

  {
    LOG_INFO( main_logger, "new scope!");

    // We can create a new instance of our track data structure...
    my_particular_track t;

    // and it's still there.
    LOG_INFO( main_logger, t( track_handle ) << "");

  }



  //
  // test the generic get-frames capability
  //
  {
    frame_handle_list_type frames = track_oracle_core::get_frames( track_handle );
    LOG_INFO( main_logger, "Generically got " << frames.size() << " frames");
  }

  //
  // test appending frames en masse
  //
  {
    // create some frames in a temp track
    my_particular_track temp_track;
    track_handle_type temp = temp_track.create();

    for (unsigned i=0; i<3; ++i)
    {
      frame_handle_type f = temp_track.create_frame();
      temp_track[f].timestamp() = 10000 + (1000 * i);
    }

    my_particular_track t;
    LOG_INFO( main_logger, "Before frame appending: " << t(track_handle) << "");
    t(track_handle).add_frames( track_oracle_core::get_frames( temp ));
    LOG_INFO( main_logger, "After frame appending: " << t(track_handle) << "");
  }

  // we can examine what fields a track or frame has
  {
    vector< field_handle_type > columns = track_oracle_core::fields_at_row( 1001 );
    LOG_INFO( main_logger, "Row 1001 has " << columns.size() << " fields:");
    for (size_t i=0; i<columns.size(); ++i)
    {
      element_descriptor e = track_oracle_core::get_element_descriptor( columns[i] );
      LOG_INFO( main_logger, i << ": " << e.name << " ( " << e.description << " ) type "
               << e.typeid_str << " role: " << element_descriptor::role2str( e.role ) << "");
    }
    LOG_INFO( main_logger, "");
  }

  // we can do lookups

  {
    my_particular_track t;
    // this is an example of when it would be nice for track_fields to know if they're associated with Track or Frame
    {
      LOG_INFO( main_logger, "Lookup main_logger, #1:");
      LOG_INFO( main_logger, "" );
      frame_handle_type lookup_handle = frame_handle_type( t.timestamp.lookup( 103, ::kwiver::track_oracle::DOMAIN_ALL ) );
      if ( lookup_handle.is_valid() )
      {
        track_oracle_frame_view f = t[ lookup_handle ].frame();
        LOG_INFO( main_logger, f << "");
        // this should no longer compile!  AND IT DOESN'T!
        //LOG_INFO( main_logger, "timestamp 103: " << f( lookup_handle ) << "");
        LOG_INFO( main_logger, "timestamp 103: " << f[ lookup_handle ] << "");
      }
      else
      {
        LOG_ERROR( main_logger, "Coudln't find timestamp 103?");
      }
    }

    {
      LOG_INFO( main_logger, "Lookup #3:");
      LOG_INFO( main_logger, "" );
      // If the track_field is associated with the Frame, we can look it up relative
      // to a track
      LOG_INFO( main_logger, "Lookup relative to track handle");
      frame_handle_type lookup_handle = frame_handle_type( t.timestamp.lookup( 103, track_handle ));
      if ( lookup_handle.is_valid() )
      {
        track_oracle_frame_view f = t[ lookup_handle].frame();
        LOG_INFO( main_logger, f << "");
        LOG_INFO( main_logger, "timestamp 103: " << f[lookup_handle] << "");
      }
      else
      {
        LOG_ERROR( main_logger, "Couldn't find timestamp 103 relative to track handle?");
      }
    }

    {
      LOG_INFO( main_logger, "Lookup #4:");
      LOG_INFO( main_logger, "" );
      // trying out data terms
      track_field< ::kwiver::track_oracle::dt::tracking::timestamp_usecs > frame_term;
      frame_handle_type lookup( frame_term.lookup( 103, ::kwiver::track_oracle::DOMAIN_ALL ));
      LOG_INFO( main_logger, "Track field lookup on data_term: timestamp 103 found? " << lookup.is_valid() );
      track_oracle_frame_view f = t[ lookup ].frame();
      LOG_INFO( main_logger, "Timestamp 103: " << f[ lookup ] );
    }
  }


  // Note that we can get to individual (handle, field-name) tuples
  // via a different track data structure:
  //

  {

    a_different_track t2;

    // TRACK HANDLES ARE TYPELESS WITH REGARD TO THE UNDERLYING TRACK DATA
    // STRUCTURE!  This is because tracks exist only as groupings of track
    // fields.  The data exists regardless of the particular grouping.

    LOG_INFO( main_logger, "A different track structure sees:\n" << t2( track_handle ) << "");

    // note that t2 also has five frames, but since they have no fields
    // associated with them, they're empty from perspective of the t2 definition.
    // Hmm.
  }

  // deleting the track is the same as removing all the fields.
  {
    my_particular_track t;
    t( track_handle).remove_me();
  }
  {
    my_particular_track t;
    LOG_INFO( main_logger, "After track removal: " << t( track_handle ) << "");
  }

  //
  // Hmm: this can't be caught at compile time; there's a runtime
  // check.  This is now caught when track_field is initialized,
  // rather than when it's used.

  LOG_INFO( main_logger, "About to define a track field which re-types 'label'...");
  try
  {
    track_field<int> conflict( "label" );
  }
  catch (runtime_error& e)
  {
    LOG_INFO( main_logger, e.what() << "");
  }

  //
  // schema comparisons
  //

  // "my_particular_track" has scores, labels, and timestamps.
  // "a_different_track" only has labels.
  // If you examine a track populated from "a_different_track"
  // and ask what fields from "my_particular_track" it's missing,
  // this will return "score" and "timestamp".
  //
  // Note that we can only verify tracks or frames if they're
  // instantiated.  If you don't call 'adt(adt_handle).create_frame()'
  // to set an empty frame on adt_handle, then the Frame row_view is
  // never checked.  This points out the sort of weird limbo inhabited
  // by the "tracks / frames" structural constraint.
  //

  {
    a_different_track adt;
    my_particular_track mpt;

    track_handle_type adt_handle = adt.create();
    adt( adt_handle ).activity_label() = "testing";
    adt( adt_handle ).create_frame(); // see comment above

    track_handle_list_type adt_tracks;

    adt_tracks.push_back( adt_handle );
    vector< element_descriptor > missing_fields =
      ::kwiver::track_oracle::schema_algorithm::name_missing_fields( mpt, adt_tracks );
    LOG_INFO( main_logger, "a_different_track schema is missing " << missing_fields.size()
              << " fields vs. my_particular_track:" );
    for (size_t i=0; i<missing_fields.size(); ++i)
    {
      LOG_INFO( main_logger, i << ": '" << missing_fields[i].name << "'" );
    }
  }

  {
    // demonstrating inherited schemata

    LOG_INFO( main_logger, "About to create an instance of a schema with a duplicated field; should throw..." );
    // this should throw, because "score" is duplicated
    try
    {
      derived_track_throws_exception dte;
    }
    catch (runtime_error& e)
    {
      LOG_INFO( main_logger, e.what() );
    }

    my_particular_track mpt;
    derived_track dt;

    track_handle_type dth = dt.create();
    dt( dth ).score() = 5.0;
    dt( dth ).foo() = 10.0;

    LOG_INFO( main_logger, "Schema view at base class:\n" << mpt( dth ) );
    LOG_INFO( main_logger, "Schema view at derived class:\n" << dt( dth ) );

  }

  LOG_INFO( main_logger, "All done" );
}
