// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_VISCL_UTILS_H_
#define KWIVER_ARROWS_VISCL_UTILS_H_

#include <vital/types/feature_set.h>

namespace kwiver {
namespace arrows {
namespace vcl {

/// Compute image dimensions from feature set
void min_image_dimensions(const vital::feature_set &feat, unsigned int &width, unsigned int &height);

} // end namespace vcl
} // end namespace arrows
} // end namespace kwiver

#endif
