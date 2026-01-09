/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Split an object track set into a feature_track_set and a landmark_map
 */

#ifndef VIAME_CORE_SPLIT_TRACKS_TO_FEATURE_LANDMARKS_PROCESS_H
#define VIAME_CORE_SPLIT_TRACKS_TO_FEATURE_LANDMARKS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "viame_processes_core_export.h"

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Split an object track set into a feature_track_set and a landmark_map
 */
class VIAME_PROCESSES_CORE_NO_EXPORT split_tracks_to_feature_landmarks_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  split_tracks_to_feature_landmarks_process( kwiver::vital::config_block_sptr const& config );
  virtual ~split_tracks_to_feature_landmarks_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class split_tracks_to_feature_landmarks_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_SPLIT_TRACKS_TO_FEATURE_LANDMARKS_PROCESS_H
