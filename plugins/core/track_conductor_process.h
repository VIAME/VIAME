/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Consolidate and control the output of multiple object trackers
 */

#ifndef VIAME_CORE_TRACK_CONDUCTOR_PROCESS_H
#define VIAME_CORE_TRACK_CONDUCTOR_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "viame_processes_core_export.h"

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Consolidate and control the output of multiple object trackers
 * 
 * The output of multiple trackers can be combined via multiple methods
 */
class VIAME_PROCESSES_CORE_NO_EXPORT track_conductor_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  track_conductor_process( kwiver::vital::config_block_sptr const& config );
  virtual ~track_conductor_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void sync_step();
  void async_step();

  void make_ports();
  void make_config();
  void make_threads();

  void wait_for_standard_inputs();
  void wait_for_short_term_inputs();
  void wait_for_mid_term_inputs();
  void wait_for_long_term_inputs();

  class priv;
  const std::unique_ptr<priv> d;
  friend class priv;

}; // end class track_conductor_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_TRACK_CONDUCTOR_PROCESS_H
