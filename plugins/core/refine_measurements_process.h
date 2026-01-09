/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Refine measurements in object detections via multiple methods
 */

#ifndef VIAME_CORE_REFINE_MEASUREMENTS_PROCESS_H
#define VIAME_CORE_REFINE_MEASUREMENTS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "viame_processes_core_export.h"

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Filter object tracks
 */
class VIAME_PROCESSES_CORE_NO_EXPORT refine_measurements_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  refine_measurements_process( kwiver::vital::config_block_sptr const& config );
  virtual ~refine_measurements_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class refine_measurements_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_REFINE_MEASUREMENTS_PROCESS_H
