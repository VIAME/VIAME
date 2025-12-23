/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Detect shot breaks and create tracks for each shot
 */

#ifndef VIAME_CORE_DETECT_SHOT_BREAKS_PROCESS_H
#define VIAME_CORE_DETECT_SHOT_BREAKS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/core/viame_processes_core_export.h>

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Detect shot breaks and create tracks for each shot
 */
class VIAME_PROCESSES_CORE_NO_EXPORT detect_shot_breaks_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  detect_shot_breaks_process( kwiver::vital::config_block_sptr const& config );
  virtual ~detect_shot_breaks_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class detect_shot_breaks_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_DETECT_SHOT_BREAKS_PROCESS_H
