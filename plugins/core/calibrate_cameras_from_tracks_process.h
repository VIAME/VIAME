/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Calibrate two cameras from two objects track set
 */

#ifndef VIAME_CORE_CALIBRATE_CAMERAS_FROM_TRACKS_PROCESS_H
#define VIAME_CORE_CALIBRATE_CAMERAS_FROM_TRACKS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "viame_processes_core_export.h"

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Calibrate two cameras from two objects track set
 */
class VIAME_PROCESSES_CORE_NO_EXPORT calibrate_cameras_from_tracks_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  calibrate_cameras_from_tracks_process( kwiver::vital::config_block_sptr const& config );
  virtual ~calibrate_cameras_from_tracks_process();

protected:
  void _configure() override;
  void _step() override;

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class calibrate_cameras_from_tracks_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_CALIBRATE_CAMERAS_FROM_TRACKS_PROCESS_H
