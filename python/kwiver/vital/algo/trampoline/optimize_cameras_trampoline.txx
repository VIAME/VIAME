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
 * \file optimize_cameras_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<optimize_cameras> and optimize_cameras
 */

#ifndef OPTIMIZE_CAMERAS_TRAMPOLINE_TXX
#define OPTIMIZE_CAMERAS_TRAMPOLINE_TXX


#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/optimize_cameras.h>


template < class algorithm_def_oc_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::optimize_cameras > >
class algorithm_def_oc_trampoline :
      public algorithm_trampoline<algorithm_def_oc_base>
{
  public:
    using algorithm_trampoline<algorithm_def_oc_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::optimize_cameras>,
        type_name,
      );
    }
};


template< class optimize_cameras_base=
                kwiver::vital::algo::optimize_cameras >
class optimize_cameras_trampoline :
      public algorithm_def_oc_trampoline< optimize_cameras_base >
{
  public:
    using algorithm_def_oc_trampoline< optimize_cameras_base>::
              algorithm_def_oc_trampoline;

    void
    optimize( kwiver::vital::camera_map_sptr& cameras,
              kwiver::vital::feature_track_set_sptr tracks,
              kwiver::vital::landmark_map_sptr landmarks,
              kwiver::vital::sfm_constraints_sptr constraints ) const override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::optimize_cameras,
        optimize,
        cameras,
        tracks,
        landmarks,
        constraints
      );
    }

    void
    optimize( kwiver::vital::camera_perspective_sptr& camera,
              std::vector< kwiver::vital::feature_sptr > const& features,
              std::vector< kwiver::vital::landmark_sptr > const& landmarks,
              kwiver::vital::sfm_constraints_sptr constraints ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::optimize_cameras,
        optimize,
        camera,
        features,
        landmarks,
        constraints
      );
    }

};

#endif
