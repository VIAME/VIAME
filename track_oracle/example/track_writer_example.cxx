/*ckwg +5
 * Copyright 2013-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

///
/// An example program demonstrating track writing.
///

#include <track_oracle/core/track_oracle_core.h>
#include <track_oracle/file_formats/file_format_base.h>
#include <track_oracle/file_formats/file_format_manager.h>


#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::cout;
using namespace kwiver::track_oracle;

int main(int argc, char *argv[])
{
  if (argc < 3)
  {
    LOG_INFO( main_logger,"Usage: " << argv[0] << " input_file format [output_file]\n\n" <<
             "Attempts to load track data from <input_file> and write it out\n"
             "again using <format> to <output_file>. If not <output_file> is\n"
             "specified, use stdout. Note that writing to stdout requires\n"
             "that <format> supports writing to streams, while specifying\n"
             "an <output_file> requires that <format> supports writing to a\n"
             "file.");
    return EXIT_FAILURE;
  }

  track_handle_list_type tracks;
  if (!file_format_manager::read(argv[1], tracks))
  {
    LOG_ERROR( main_logger,"Error: could not read tracks from '" << argv[1] << '\'');
    return EXIT_FAILURE;
  }

  file_format_enum const ff_type = file_format_type::from_string(argv[2]);
  if (ff_type == TF_INVALID_TYPE)
  {
    LOG_ERROR( main_logger,"Error: could not find format '" << argv[2] << '\'');
    return EXIT_FAILURE;
  }

  file_format_base* const ff = file_format_manager::get_format(ff_type);
  if (!ff)
  {
    LOG_ERROR( main_logger,"Error: failed to load file format manager for '" << argv[2] << '\'');
    return EXIT_FAILURE;
  }

  if (argc > 3)
  {
    if (!ff->write(argv[3], tracks))
    {
      LOG_ERROR( main_logger,"Error: failed to write tracks to '" << argv[3] << '\'');
      return EXIT_FAILURE;
    }
  }
  else
  {
    if (!ff->write(cout, tracks))
    {
      LOG_ERROR( main_logger,"Error: failed to write tracks to stdout");
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
