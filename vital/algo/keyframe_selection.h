// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_ALGO_KEYFRAME_SELECTION_H_
#define VITAL_ALGO_KEYFRAME_SELECTION_H_

#include <vital/vital_config.h>

#include <utility>
#include <vector>
#include <memory>

#include <vital/algo/algorithm.h>
#include <vital/types/track_set.h>

/**
* \file
* \brief Header defining abstract \link kwiver::vital::algo::keyframe_selection
*        keyframe selection \endlink algorithm
*/

namespace kwiver {
namespace vital {
namespace algo {

  /// \brief Abstract base class for track set filter algorithms.
  class VITAL_ALGO_EXPORT keyframe_selection
    : public kwiver::vital::algorithm_def<keyframe_selection>
  {
  public:

    /// Return the name of this algorithm.
    static std::string static_type_name() { return "keyframe_selection"; }

    /// Select keyframes from a set of tracks.
    /** Different implementations can select key-frames in different ways.
    *   For example, one method could only add key-frames for frames that are new.  Another could increase the
    * density of key-frames near existing frames so dense processing can be done.
    */
    /**
    * \param [in] tracks The tracks over which to select key-frames
    * \returns a track set that includes the selected keyframe data structure
    */
    virtual kwiver::vital::track_set_sptr
      select(kwiver::vital::track_set_sptr tracks) const = 0;
  protected:

    /// Default constructor
    keyframe_selection();
  };

  /// type definition for shared pointer to a filter_tracks algorithm
  typedef std::shared_ptr<keyframe_selection> keyframe_selection_sptr;

}}} // end namespace

#endif // VITAL_ALGO_KEYFRAME_SELECTION_H_
