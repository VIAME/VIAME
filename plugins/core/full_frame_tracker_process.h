/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Link full frame detections into full frame tracks
 */

#ifndef VIAME_CORE_FULL_FRAME_TRACKER_PROCESS_H
#define VIAME_CORE_FULL_FRAME_TRACKER_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/core/viame_processes_core_export.h>

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Link full frame detections into full frame tracks
 */
class VIAME_PROCESSES_CORE_NO_EXPORT full_frame_tracker_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  full_frame_tracker_process( kwiver::vital::config_block_sptr const& config );
  virtual ~full_frame_tracker_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class full_frame_tracker_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_FULL_FRAME_TRACKER_PROCESS_H
