/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief CSV descriptor storage backend for sprokit processes
 */

#ifndef VIAME_FILE_IO_STORE_DESCRIPTORS_CSV_H
#define VIAME_FILE_IO_STORE_DESCRIPTORS_CSV_H

#include "viame_processes_file_io_export.h"

#include <viame/core_types/viame_core_types.h>

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief CSV file backend for descriptor storage
 *
 * This class handles reading and writing descriptors to CSV files.
 * Format: uid,val1,val2,...,valN (one descriptor per line)
 */
class VIAME_PROCESSES_FILE_IO_EXPORT csv_descriptor_backend
{
public:
  csv_descriptor_backend( const std::string& file_path );
  ~csv_descriptor_backend();

  // Write operations
  void open_for_write( bool append = false );
  void write_descriptor( const std::string& uid,
                         const std::vector< double >& values );
  void flush();
  void close();

  // Read operations
  void load_index();
  bool get_descriptor( const std::string& uid,
                       std::vector< double >& values );
  bool get_descriptor_by_track_frame(
    viame::track_id_t track_id,
    viame::frame_id_t frame_id,
    std::vector< double >& values );

  // Additional CSV-specific configuration
  void set_uid_mapping_file( const std::string& path );
  void set_track_frame_file( const std::string& path );

private:
  class impl;
  std::unique_ptr< impl > p;
};

} // end namespace core
} // end namespace viame

#endif // VIAME_FILE_IO_STORE_DESCRIPTORS_CSV_H
