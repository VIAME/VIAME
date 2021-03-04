// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file bundle_adjust_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<bundle_adjust> and bundle_adjust
 */

#ifndef BUNDLE_ADJUST_TRAMPOLINE_TXX
#define BUNDLE_ADJUST_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/bundle_adjust.h>

namespace kwiver {
namespace vital  {
namespace python {

template <class algorithm_def_ba_base=kwiver::vital::algorithm_def<kwiver::vital::algo::bundle_adjust>>
class algorithm_def_ba_trampoline :
      public algorithm_trampoline<algorithm_def_ba_base>
{
  public:
    using algorithm_trampoline<algorithm_def_ba_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::bundle_adjust>,
        type_name,
      );
    }
};

template <class bundle_adjust_base=kwiver::vital::algo::bundle_adjust>
class bundle_adjust_trampoline :
      public algorithm_def_ba_trampoline<bundle_adjust_base>
{
  public:
    using algorithm_def_ba_trampoline<bundle_adjust_base>::
              algorithm_def_ba_trampoline;

    void optimize( kwiver::vital::camera_map_sptr& cameras,
                   kwiver::vital::landmark_map_sptr& landmarks,
                   kwiver::vital::feature_track_set_sptr tracks,
                   kwiver::vital::sfm_constraints_sptr constraints=nullptr) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::bundle_adjust,
        optimize,
        cameras,
        landmarks,
        tracks,
        constraints
      );
    }

    void optimize( kwiver::vital::simple_camera_perspective_map& cameras,
                   kwiver::vital::landmark_map::map_landmark_t& landmarks,
                   kwiver::vital::feature_track_set_sptr tracks,
                   const std::set<kwiver::vital::frame_id_t>& fixed_cameras,
                   const std::set<kwiver::vital::landmark_id_t>& fixed_landmarks,
                   kwiver::vital::sfm_constraints_sptr constraints=nullptr) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::bundle_adjust,
        optimize,
        cameras,
        landmarks,
        tracks,
        fixed_cameras,
        fixed_landmarks,
        constraints
      );
    }
};
}
}
}
#endif
