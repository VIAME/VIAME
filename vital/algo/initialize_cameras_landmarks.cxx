// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Instantiation of \link kwiver::vital::algo::algorithm_def algorithm_def<T> \endlink
 *        for \link kwiver::vital::algo::initialize_cameras_landmarks
 *        initialize_cameras_landmarks \endlink
 */

#include <vital/algo/initialize_cameras_landmarks.h>
#include <vital/algo/algorithm.txx>

namespace kwiver {
namespace vital {
namespace algo {

initialize_cameras_landmarks
::initialize_cameras_landmarks()
{
  attach_logger( "algo.initialize_cameras_landmarks" );
}

/// Set a callback function to report intermediate progress
void
initialize_cameras_landmarks
::set_callback(callback_t cb)
{
  this->m_callback = cb;
}

} } }

/// \cond DoxygenSuppress
  INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::initialize_cameras_landmarks);
/// \endcond
