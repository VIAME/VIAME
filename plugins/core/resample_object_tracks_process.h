/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Resample object tracks from one downsample rate to another
 */

#ifndef VIAME_CORE_RESAMPLE_OBJECT_TRACKS_PROCESS_H
#define VIAME_CORE_RESAMPLE_OBJECT_TRACKS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "viame_processes_core_export.h"

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Resample object tracks from one downsample rate to another
 *
 * Loads tracks from a VIAME CSV file at configure time, receives timestamps
 * from a video source at the output rate, and outputs tracks with linearly
 * interpolated bounding boxes for intermediate frames.
 */
class VIAME_PROCESSES_CORE_NO_EXPORT resample_object_tracks_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  resample_object_tracks_process( kwiver::vital::config_block_sptr const& config );
  virtual ~resample_object_tracks_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class resample_object_tracks_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_RESAMPLE_OBJECT_TRACKS_PROCESS_H
