/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
