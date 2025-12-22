/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Align multi modal images that may be temporally out of sync
 */

#ifndef VIAME_CORE_FILTER_FRAME_PROCESS_H
#define VIAME_CORE_FILTER_FRAME_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/core/viame_processes_core_export.h>

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Filters out frames if certain criteria are not met
 */
class VIAME_PROCESSES_CORE_NO_EXPORT filter_frame_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  filter_frame_process( kwiver::vital::config_block_sptr const& config );
  virtual ~filter_frame_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class filter_frame_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_FILTER_FRAME_PROCESS_H
