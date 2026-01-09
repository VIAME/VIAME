/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Select between two database query inputs
 */

#ifndef VIAME_CORE_SELECT_DATABASE_QUERY_PROCESS_H
#define VIAME_CORE_SELECT_DATABASE_QUERY_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "viame_processes_core_export.h"

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Selects between two database_query inputs
 *
 * This process takes two database_query inputs (primary and fallback) and
 * outputs the primary if it's non-null, otherwise outputs the fallback.
 * Useful for conditionally routing auto-generated queries vs external queries.
 */
class VIAME_PROCESSES_CORE_NO_EXPORT select_database_query_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  select_database_query_process( kwiver::vital::config_block_sptr const& config );
  virtual ~select_database_query_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class select_database_query_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_SELECT_DATABASE_QUERY_PROCESS_H
