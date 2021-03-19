// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Interface for detected_object_set_output_csv

#ifndef KWIVER_ARROWS_SERIALIZE_METADATA_CSV_H
#define KWIVER_ARROWS_SERIALIZE_METADATA_CSV_H

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/serialize_metadata.h>

namespace kwiver {

namespace arrows {

namespace core {

class KWIVER_ALGO_CORE_EXPORT serialize_metadata_csv
  : public vital::algo::serialize_metadata
{
public:
  PLUGIN_INFO( "csv",
               "Metadata map writer using CSV format." )

  serialize_metadata_csv();
  virtual ~serialize_metadata_csv();

  /// Unimplemented
  ///
  /// \param filename the path to the file the load
  /// \throws kwiver::vital::file_write_exception not implemented
  virtual kwiver::vital::metadata_map_sptr load_( std::string const& filename )
  const;

  /// Implementation specific save functionality.
  ///
  /// Save metadata to a CSV file. Uses the union of fields taken from all
  /// packets as the header, and inserts empty fields when values are missing
  /// for a given frame
  ///
  /// \param filename the path to the file to save
  /// \param data the metadata for a video to save
  virtual void save_( std::string const& filename,
                      kwiver::vital::metadata_map_sptr data ) const;

private:
  class priv;

  std::unique_ptr< priv > d_;
};

} // namespace core

} // namespace arrows

} // namespace kwiver

#endif
