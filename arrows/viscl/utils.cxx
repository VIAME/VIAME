// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "utils.h"

namespace kwiver {
namespace arrows {
namespace vcl {

/// Compute image dimensions from feature set
void min_image_dimensions(const vital::feature_set &feat, unsigned int &width, unsigned int &height)
{
  width = 0;
  height = 0;

  std::vector<vital::feature_sptr> features = feat.features();
  for (unsigned int i = 0; i < features.size(); i++)
  {
    if (width < features[i]->loc()[0])
    {
      width = features[i]->loc()[0];
    }

    if (height < features[i]->loc()[1])
    {
      height = features[i]->loc()[1];
    }
  }
}

} // end namespace vcl
} // end namespace arrows
} // end namespace kwiver
