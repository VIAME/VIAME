/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Fetch descriptors from database given UIDs
 */

#ifndef VIAME_CPPDB_FETCH_DESCRIPTORS_DB_PROCESS_H
#define VIAME_CPPDB_FETCH_DESCRIPTORS_DB_PROCESS_H

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
 * @brief Fetch descriptors from database given UIDs
 *
 * This process takes in a vector of UIDs and fetches the corresponding
 * descriptors from a database. This is a C++ replacement for the SMQTK-based
 * smqtk_fetch_descriptors Python process with database backend.
 */
class VIAME_PROCESSES_CPPDB_NO_EXPORT fetch_descriptors_db_process
  : public sprokit::process
{
public:
  using config_block_sptr = kwiver::vital::config_block_sptr;

  // -- CONSTRUCTORS --
  fetch_descriptors_db_process( config_block_sptr const& config );
  virtual ~fetch_descriptors_db_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr< priv > d;

}; // end class fetch_descriptors_db_process

} // end namespace cppdb
} // end namespace viame

#endif // VIAME_CPPDB_FETCH_DESCRIPTORS_DB_PROCESS_H
