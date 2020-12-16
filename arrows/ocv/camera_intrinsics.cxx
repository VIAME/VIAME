// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV camera intrinsics implementation
 */

#include "camera_intrinsics.h"

namespace kwiver {
namespace arrows {
namespace ocv {

std::vector<double> dist_coeffs_to_ocv(std::vector<double> const& vital_dist_coeffs)
{
  std::vector<double> ocv_dist;
  size_t num_coeffs = vital_dist_coeffs.size() < 4 ? 4 : vital_dist_coeffs.size();
  ocv_dist.assign(num_coeffs, 0);
  for (unsigned int i = 0; i < vital_dist_coeffs.size(); ++i)
  {
    ocv_dist[i] = vital_dist_coeffs[i];
  }

  return ocv_dist;
}

std::vector<double> get_ocv_dist_coeffs(vital::camera_intrinsics_sptr intrinsics)
{
  return dist_coeffs_to_ocv(intrinsics->dist_coeffs());
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
