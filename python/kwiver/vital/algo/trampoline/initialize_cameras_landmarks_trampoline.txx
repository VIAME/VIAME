// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file initialize_cameras_landmarks_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<initialize_cameras_landmarks> and initialize_cameras_landmarks
 */

#ifndef INITIALIZE_CAMERAS_LANDMARKS_TRAMPOLINE_TXX
#define INITIALIZE_CAMERAS_LANDMARKS_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/initialize_cameras_landmarks.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_icl_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::initialize_cameras_landmarks > >
class algorithm_def_icl_trampoline :
      public algorithm_trampoline<algorithm_def_icl_base>
{
  public:
    using algorithm_trampoline<algorithm_def_icl_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::initialize_cameras_landmarks>,
        type_name,
      );
    }
};

template< class initialize_cameras_landmarks_base=
                kwiver::vital::algo::initialize_cameras_landmarks >
class initialize_cameras_landmarks_trampoline :
      public algorithm_def_icl_trampoline< initialize_cameras_landmarks_base >
{
  public:
    using algorithm_def_icl_trampoline< initialize_cameras_landmarks_base>::
              algorithm_def_icl_trampoline;

    void
    initialize( kwiver::vital::camera_map_sptr& cameras,
               kwiver::vital::landmark_map_sptr& landmarks,
               kwiver::vital::feature_track_set_sptr tracks,
               kwiver::vital::sfm_constraints_sptr constraints ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::initialize_cameras_landmarks,
        initialize,
        cameras,
        landmarks,
        tracks,
        constraints
      );
    }
};

}
}
}

#endif
