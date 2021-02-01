// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

///
/// An example program demonstrating querying for a dynamically selected attribute.
/// This example never uses knowledge of the track format, although it could.
///

#include <iostream>
#include <map>
#include <string>
#include <cstdlib>

#include <track_oracle/core/track_oracle_core.h>
#include <track_oracle/core/element_descriptor.h>
#include <track_oracle/core/track_field.h>
#include <track_oracle/file_formats/file_format_manager.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::string;

using namespace kwiver::track_oracle;

int main( int argc, char *argv[] )
{
  if (argc != 3)
  {
    LOG_INFO( main_logger, "Usage: " << argv[0] << " track-file field_name\n"
             << "Load in a track file, attempt to see if it supplies field_name.\n"
             << "(Type is assumed to be double.)");
    return EXIT_FAILURE;
  }

  // try to load the tracks
  const string track_fn( argv[1] );
  track_handle_list_type tracks;
  if (! file_format_manager::read( track_fn, tracks ))
  {
    LOG_ERROR( main_logger, "Error: couldn't read tracks from '" << track_fn << "'; exiting");
    return EXIT_FAILURE;
  }
  LOG_INFO( main_logger, "Info: read " << tracks.size() << " tracks");
  if ( tracks.empty() )
  {
    LOG_INFO( main_logger, "Info: reader succeeded but no tracks were loaded?  Weird!");
    return EXIT_FAILURE;
  }

  // Create a track field for the user-specified name
  track_field<double> user_field( argv[2] );

  // our example operation on the user-specified field will be...
  // ... (drum roll) ...
  // ... take the average of it!  BOOOORRRRING yes, but you get the idea.

  double sum = 0.0;
  size_t count = 0;

  // This approach is conservative, in that each individual track and
  // frame is queried to see if the user_field exists... As soon as
  // we find it once, we could remember if it was on the track-level-data
  // or frame-level-data, but here we just blindly ask each time.

  for (size_t t=0; t<tracks.size(); ++t)
  {
    if (user_field.exists( tracks[t].row ))
    {
      sum += user_field( tracks[t].row );
      ++count;
    }

    frame_handle_list_type frames = track_oracle_core::get_frames( tracks[t] );
    for (size_t f=0; f<frames.size(); ++f)
    {
      if (user_field.exists( frames[f].row ))
      {
        sum += user_field( frames[f].row );
        ++count;
      }
    }
  }

  // all done

  LOG_INFO( main_logger, "Info: found " << count << " instances of '" << argv[2] << "' in the file");
  if (count > 0)
  {
    LOG_INFO( main_logger, "Info: ... the sum was " << sum << " and the average was " << sum / count << "");
  }

}
