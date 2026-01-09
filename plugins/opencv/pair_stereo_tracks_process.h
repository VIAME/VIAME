/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Compute object tracks pair from stereo depth map information
 */

#ifndef VIAME_OPENCV_PAIR_STEREO_TRACKS_PROCESS_H
#define VIAME_OPENCV_PAIR_STEREO_TRACKS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "viame_processes_opencv_export.h"

#include <memory>

namespace viame
{

class pair_stereo_tracks;

// -----------------------------------------------------------------------------
/**
 * @brief Compute object tracks pair from stereo depth map information
 */
class VIAME_PROCESSES_OPENCV_NO_EXPORT pair_stereo_tracks_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  pair_stereo_tracks_process( kwiver::vital::config_block_sptr const& config );
  virtual ~pair_stereo_tracks_process();

protected:
  void _configure() override;
  void _step() override;

private:
  void make_ports();
  void make_config();

  const std::unique_ptr<pair_stereo_tracks> d;

}; // end class pair_stereo_tracks_process

} // end namespace viame

#endif // VIAME_OPENCV_PAIR_STEREO_TRACKS_PROCESS_H
