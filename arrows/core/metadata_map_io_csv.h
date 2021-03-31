// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Interface for detected_object_set_output_csv

#ifndef KWIVER_ARROWS_METADATA_MAP_IO_CSV_H
#define KWIVER_ARROWS_METADATA_MAP_IO_CSV_H

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/metadata_map_io.h>

#include <iostream>

namespace kwiver {

namespace arrows {

namespace core {

class KWIVER_ALGO_CORE_EXPORT metadata_map_io_csv
  : public vital::algo::metadata_map_io
{
public:
  PLUGIN_INFO( "csv",
               "Metadata map writer using CSV format." )

  metadata_map_io_csv();
  virtual ~metadata_map_io_csv();

  /// Unimplemented.
  ///
  /// \param filename the path to the file the load
  /// \throws kwiver::vital::file_write_exception not implemented
  kwiver::vital::metadata_map_sptr load_(
    std::istream& fin, std::string const& filename ) const override;

  /// Implementation specific save functionality.
  ///
  /// Save metadata to a CSV file. Uses the union of fields taken from all
  /// packets as the header, and inserts empty fields when values are missing
  /// for a given frame
  ///
  /// \param filename the path to the file to save
  /// \param data the metadata for a video to save
  void save_( std::ostream& fout,
              kwiver::vital::metadata_map_sptr data,
              std::string const& filename ) const override;

private:
  class priv;

  std::unique_ptr< priv > d_;
};

} // namespace core

} // namespace arrows

} // namespace kwiver

#endif
