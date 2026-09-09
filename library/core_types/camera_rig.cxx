// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation for \link kwiver::vital::camera_rig_stereo
/// camera_rig_stereo \endlink

#include "camera_rig.h"

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
camera_rig_stereo
::camera_rig_stereo( camera_sptr left, camera_sptr right )
{
  add( "left", left );
  add( "right", right );
}

// ----------------------------------------------------------------------------
camera_rig_stereo
::~camera_rig_stereo()
{}

} // vital

} // kwiver
