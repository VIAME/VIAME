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
 * \file estimate_similarity_transform_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::estimate_similarity_transform \endlink
 */

#ifndef ESTIMATE_SIMILARITY_TRANSFORM_TRAMPOLINE_TXX
#define ESTIMATE_SIMILARITY_TRANSFORM_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/estimate_similarity_transform.h>

namespace kwiver {
namespace vital  {
namespace python {
template< class algorithm_def_est_base=
           kwiver::vital::algorithm_def<
               kwiver::vital::algo::estimate_similarity_transform > >
class algorithm_def_est_trampoline :
      public algorithm_trampoline< algorithm_def_est_base>
{
  public:
    using algorithm_trampoline< algorithm_def_est_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<
              kwiver::vital::algo::estimate_similarity_transform >,
        type_name,
      );
    }
};


template< class estimate_similarity_transform_base =
                  kwiver::vital::algo::estimate_similarity_transform >
class estimate_similarity_transform_trampoline :
      public algorithm_def_est_trampoline< estimate_similarity_transform_base >
{
  public:
    using algorithm_def_est_trampoline< estimate_similarity_transform_base >::
              algorithm_def_est_trampoline;

    kwiver::vital::similarity_d
      estimate_transform( std::vector< kwiver::vital::vector_3d >  const& from,
                          std::vector< kwiver::vital::vector_3d >  const& to
                        )  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::similarity_d,
        kwiver::vital::algo::estimate_similarity_transform,
        estimate_transform,
        from,
        to
      );
    }

    kwiver::vital::similarity_d
      estimate_transform(
          std::vector< kwiver::vital::camera_perspective_sptr >  const& from,
          std::vector< kwiver::vital::camera_perspective_sptr >  const& to
          )  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::similarity_d,
        kwiver::vital::algo::estimate_similarity_transform,
        estimate_transform,
        from,
        to
      );
    }

    kwiver::vital::similarity_d
      estimate_transform(
          std::vector< kwiver::vital::landmark_sptr >  const& from,
          std::vector< kwiver::vital::landmark_sptr >  const& to
          )  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::similarity_d,
        kwiver::vital::algo::estimate_similarity_transform,
        estimate_transform,
        from,
        to
      );
    }

    kwiver::vital::similarity_d
      estimate_transform(
          kwiver::vital::camera_map_sptr const from,
          kwiver::vital::camera_map_sptr const to
          )  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::similarity_d,
        kwiver::vital::algo::estimate_similarity_transform,
        estimate_transform,
        from,
        to
      );
    }

    kwiver::vital::similarity_d
      estimate_transform(
          kwiver::vital::landmark_map_sptr const from,
          kwiver::vital::landmark_map_sptr const to
          )  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::similarity_d,
        kwiver::vital::algo::estimate_similarity_transform,
        estimate_transform,
        from,
        to
      );
    }
};
}
}
}

#endif
