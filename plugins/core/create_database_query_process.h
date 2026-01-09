/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Create a database query from track descriptors
 */

#ifndef VIAME_CORE_CREATE_DATABASE_QUERY_PROCESS_H
#define VIAME_CORE_CREATE_DATABASE_QUERY_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "viame_processes_core_export.h"

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Creates a database_query from track_descriptor_set
 *
 * This process takes a track_descriptor_set and creates a database_query
 * object that can be used with the perform_query process for similarity
 * searches in the indexed database.
 */
class VIAME_PROCESSES_CORE_NO_EXPORT create_database_query_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  create_database_query_process( kwiver::vital::config_block_sptr const& config );
  virtual ~create_database_query_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class create_database_query_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_CREATE_DATABASE_QUERY_PROCESS_H
