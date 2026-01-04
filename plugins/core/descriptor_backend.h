/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief CSV descriptor storage backend for sprokit processes
 */

#ifndef VIAME_CORE_DESCRIPTOR_BACKEND_H
#define VIAME_CORE_DESCRIPTOR_BACKEND_H

#include "viame_processes_core_export.h"

#include <vital/vital_types.h>

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
class VIAME_PROCESSES_CORE_NO_EXPORT csv_descriptor_backend
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
    kwiver::vital::track_id_t track_id,
    kwiver::vital::frame_id_t frame_id,
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

#endif // VIAME_CORE_DESCRIPTOR_BACKEND_H
