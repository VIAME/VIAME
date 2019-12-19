/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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
 * \file compute_depth_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::compute_depth \endlink
 */

#ifndef COMPUTE_DEPTH_TXX
#define COMPUTE_DEPTH_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/compute_depth.h>

template< class algorithm_def_cd_base=
            kwiver::vital::algorithm_def<
               kwiver::vital::algo::compute_depth > >
class algorithm_def_cd_trampoline :
      public algorithm_trampoline<algorithm_def_cd_base>
{
  public:
    using algorithm_trampoline< algorithm_def_cd_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<
          kwiver::vital::algo::compute_depth>,
        type_name,
      );
    }
};


template< class compute_depth_base=
                  kwiver::vital::algo::compute_depth >
class compute_depth_trampoline :
      public algorithm_def_cd_trampoline< compute_depth_base >
{
  public:
    using algorithm_def_cd_trampoline< compute_depth_base >::
              algorithm_def_cd_trampoline;

    kwiver::vital::image_container_sptr
    compute( std::vector<kwiver::vital::image_container_sptr> const& frames,
              std::vector<kwiver::vital::camera_perspective_sptr> const& cameras,
              double depth_min, double depth_max,
              unsigned int reference_frame,
              kwiver::vital::bounding_box<int> const& roi,
              std::vector<kwiver::vital::image_container_sptr> const& mask )
         const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::image_container_sptr,
        kwiver::vital::algo::compute_depth,
        compute,
        frames,
        cameras,
        depth_min,
        depth_max,
        reference_frame,
        roi,
        mask
      );
    }
};
#endif
