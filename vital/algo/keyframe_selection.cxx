// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/algo/algorithm.txx>

#include "keyframe_selection.h"

namespace kwiver {
namespace vital {
namespace algo {

      keyframe_selection
        ::keyframe_selection()
      {
        attach_logger("algo.keyframe_selection");
      }

}}} // end namespace

  /// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::keyframe_selection);
/// \endcond
