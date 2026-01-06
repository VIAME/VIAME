/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Write query results as object track CSV
 */

#ifndef VIAME_CORE_WRITE_QUERY_RESULTS_AS_TRACKS_PROCESS_H
#define VIAME_CORE_WRITE_QUERY_RESULTS_AS_TRACKS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "viame_processes_core_export.h"

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Writes query results as object track CSV
 *
 * This process takes query results (from a database query) and extracts
 * the bounding box information from the track descriptor history to write
 * an object track CSV. The relevancy score from each query result is used
 * as the detection confidence. This supports both single-frame detections
 * (as single-state tracks) and multi-frame tracks.
 */
class VIAME_PROCESSES_CORE_NO_EXPORT write_query_results_as_tracks_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  write_query_results_as_tracks_process( kwiver::vital::config_block_sptr const& config );
  virtual ~write_query_results_as_tracks_process();

protected:
  virtual void _configure();
  virtual void _init();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class write_query_results_as_tracks_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_WRITE_QUERY_RESULTS_AS_TRACKS_PROCESS_H
