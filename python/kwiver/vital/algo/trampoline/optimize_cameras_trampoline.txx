// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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

namespace kwiver {
namespace vital  {
namespace python {

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

}
}
}

#endif
