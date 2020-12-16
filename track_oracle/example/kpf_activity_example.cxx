// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file
 * @brief A quick test of reading and writing KPF activity files.
 *
 * Mostly because track_reader_example couldn't handle any more options.
 *
 */

#include <vul/vul_arg.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

#include <track_oracle/core/track_oracle_core.h>
#include <track_oracle/file_formats/file_format_manager.h>
#include <track_oracle/file_formats/file_format_base.h>
#include <track_oracle/file_formats/track_filter_kpf_activity/track_filter_kpf_activity.h>

using std::string;

using namespace kwiver::track_oracle;

int main( int argc, char *argv[] )
{
  vul_arg< string > geom_in_fn_arg( "-in-geom", "Input geometry (any format)" );
  vul_arg< string > act_in_fn_arg( "-in-act", "Input activity (KPF)" );
  vul_arg< string > output_prefix_arg( "-out", "Output prefix (will write .geom.yml, .activity.yml");
  vul_arg< int > activity_domain_arg( "-d", "Activity KPF domain", 2 );

  vul_arg_parse( argc, argv );
  file_format_manager::initialize();

  if (! (geom_in_fn_arg.set() && act_in_fn_arg.set() && output_prefix_arg.set()))
  {
    LOG_ERROR( main_logger, "Must set all three file arguments" );
    return EXIT_FAILURE;
  }

  track_handle_list_type input_geom_tracks;
  if ( ! file_format_manager::read( geom_in_fn_arg(), input_geom_tracks ))
  {
    LOG_ERROR( main_logger, "Couldn't read input geometry from '" << geom_in_fn_arg() << "'" );
    return EXIT_FAILURE;
  }
  LOG_INFO( main_logger, "Read " << input_geom_tracks.size() << " input geometry tracks" );

  track_handle_list_type activity_tracks;
  if ( ! track_filter_kpf_activity::read( act_in_fn_arg(),
                                          input_geom_tracks,
                                          activity_domain_arg(),
                                          activity_tracks ))
  {
    LOG_ERROR( main_logger, "Couldn't read activity tracks from '" << act_in_fn_arg() << "'" );
    return EXIT_FAILURE;
  }

  LOG_INFO( main_logger, "Generated " << activity_tracks.size() << " activity tracks" );

  {
    string geom_out_fn = output_prefix_arg()+".geom.yml";
    bool okay = file_format_manager::get_format( TF_KPF_GEOM )->write( geom_out_fn, input_geom_tracks );
    LOG_INFO( main_logger, "Wrote KPF geometry to " << geom_out_fn << " success: " << okay );
  }

  {
    string act_out_fn = output_prefix_arg()+".activity.yml";
    bool okay = track_filter_kpf_activity::write( act_out_fn, activity_tracks );
    LOG_INFO( main_logger, "Wrote KPF activity to " << act_out_fn << " success: " << okay );
  }
}
