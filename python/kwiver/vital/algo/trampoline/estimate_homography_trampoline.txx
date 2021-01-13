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
 * \file estimate_homography_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::estimate_homography \endlink
 */

#ifndef ESTIMATE_HOMOGRAPHY_TRAMPOLINE_TXX
#define ESTIMATE_HOMOGRAPHY_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/estimate_homography.h>
#include <vital/types/vector.h>

namespace kwiver {
namespace vital  {
namespace python {
template< class algorithm_def_eh_base=
           kwiver::vital::algorithm_def<
               kwiver::vital::algo::estimate_homography > >
class algorithm_def_eh_trampoline :
      public algorithm_trampoline< algorithm_def_eh_base>
{
  public:
    using algorithm_trampoline< algorithm_def_eh_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<
         kwiver::vital::algo::estimate_homography >,
        type_name,
      );
    }
};


template< class estimate_homography_base =
                  kwiver::vital::algo::estimate_homography >
class estimate_homography_trampoline :
      public algorithm_def_eh_trampoline< estimate_homography_base >
{
  public:
    using algorithm_def_eh_trampoline< estimate_homography_base >::
              algorithm_def_eh_trampoline;


    kwiver::vital::homography_sptr
      estimate(const kwiver::vital::feature_set_sptr feat1,
               const kwiver::vital::feature_set_sptr feat2,
               const kwiver::vital::match_set_sptr matches,
               std::vector<bool>& inliers,
               double inlier_scale)  const override
    {
      VITAL_PYBIND11_OVERLOAD(
        kwiver::vital::homography_sptr,
        kwiver::vital::algo::estimate_homography,
        estimate,
        feat1,
        feat2,
        matches,
        inliers,
        inlier_scale
      );
    }

    kwiver::vital::homography_sptr
      estimate(const std::vector<kwiver::vital::vector_2d>& pts1,
               const std::vector<kwiver::vital::vector_2d>& pts2,
               std::vector<bool>& inliers,
               double inlier_scale)  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::homography_sptr,
        kwiver::vital::algo::estimate_homography,
        estimate,
        pts1,
        pts2,
        inliers,
        inlier_scale
      );
    }

};
}
}
}

#endif
