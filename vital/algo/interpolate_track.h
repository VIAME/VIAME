// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_ALGO_INTERPOLATE_TRACK_H
#define VITAL_ALGO_INTERPOLATE_TRACK_H

#include <vital/vital_config.h>
#include <vital/types/object_track_set.h>
#include <vital/algo/algorithm.h>
#include <vital/algo/video_input.h>

namespace kwiver {
namespace vital {
namespace algo {

/// Abstract base class for interpolating track states.
/**
 * This class represents the abstract interface for algorithms that
 * interpolate track states.
 */
class VITAL_ALGO_EXPORT interpolate_track
  : public kwiver::vital::algorithm_def<interpolate_track>
{
public:
  /// Return the name of this algorithm.
  static std::string static_type_name() { return "interpolate_track"; }

  /// Supply video input algorithm
  /**
   * This method supplies the video input algorithm to use for getting
   * the frames needed to interpolate track states.
   *
   * @param input Pointer to the video input algorithm
   */
  void set_video_input( video_input_sptr input );

  /// This method interpolates the states between track states.
  /**
   * This method interpolates track states to fill in missing states
   * between the states supplied in the input parameter. An output
   * track is created that contains all states between the first and
   * last state in the intpu track.
   *
   * @param init_states List of states to interpolate between.
   *
   * @return Output track with missing states filled in.
   */
  virtual track_sptr interpolate( track_sptr init_states ) = 0;

  /// Typedef for the callback function signature
  typedef std::function<void(int, int)> progress_callback_t;

  /// Set a callback function to report intermediate progress
  /**
   * This method establishes the callback function that will be called
   * occasionally to report the progress of the interpolation
   * operation.
   *
   * @param cb The callback function.
   */
  void set_progress_callback( progress_callback_t cb );

protected:
  interpolate_track();

  /**
   * Call the supplied progress callback function if there is one
   * currently active.
   *
   * @param progress Fraction of process that is complete 0 - 1.0
   */
  void do_callback( float progress );

  /**
   * Call the supplied progress callback function if there is one
   * currently active.
   *
   * @param progress Number of "steps" completed
   * @param total Total number of "steps" for the process
   */
  void do_callback( int progress, int total );

  // Instance data
  video_input_sptr m_video_input;
  progress_callback_t m_progress_callback;
};

} } } // end namespace

#endif /* VITAL_ALGO_INTERPOLATE_TRACK_H */
