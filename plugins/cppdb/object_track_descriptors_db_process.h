/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Attach descriptors to object track states from database
 */

#ifndef VIAME_CPPDB_OBJECT_TRACK_DESCRIPTORS_DB_PROCESS_H
#define VIAME_CPPDB_OBJECT_TRACK_DESCRIPTORS_DB_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "viame_processes_cppdb_export.h"

#include <sprokit/processes/kwiver_type_traits.h>

#include <memory>

namespace viame
{

namespace cppdb
{

// -----------------------------------------------------------------------------
/**
 * @brief Attach descriptors to object track states from database
 *
 * This process takes in an object_track_set and attaches descriptors to each
 * track state by querying them from a database.
 *
 * Uses the same database schema as the original Python version:
 * - track_descriptor table: uid, type, video_name
 * - track_descriptor_track table: uid, track_id
 * - track_descriptor_history table: uid, frame_number, timestamp, bboxes
 * - DESCRIPTOR table: uid, video_name, vector_data, vector_size
 */
class VIAME_PROCESSES_CPPDB_NO_EXPORT object_track_descriptors_db_process
  : public sprokit::process
{
public:
  using config_block_sptr = kwiver::vital::config_block_sptr;

  // -- CONSTRUCTORS --
  object_track_descriptors_db_process( config_block_sptr const& config );
  virtual ~object_track_descriptors_db_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr< priv > d;

}; // end class object_track_descriptors_db_process

} // end namespace cppdb
} // end namespace viame

#endif // VIAME_CPPDB_OBJECT_TRACK_DESCRIPTORS_DB_PROCESS_H
