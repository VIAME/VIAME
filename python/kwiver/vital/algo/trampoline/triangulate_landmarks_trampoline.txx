// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file triangulate_landmarks_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<triangulate_landmarks> and triangulate_landmarks
 */

#ifndef TRIANGULATE_LANDMARKS_TRAMPOLINE_TXX
#define TRIANGULATE_LANDMARKS_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/triangulate_landmarks.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_tl_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::triangulate_landmarks > >
class algorithm_def_tl_trampoline :
      public algorithm_trampoline<algorithm_def_tl_base>
{
  public:
    using algorithm_trampoline<algorithm_def_tl_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::triangulate_landmarks>,
        type_name,
      );
    }
};

template< class triangulate_landmarks_base=
                kwiver::vital::algo::triangulate_landmarks >
class triangulate_landmarks_trampoline :
      public algorithm_def_tl_trampoline< triangulate_landmarks_base >
{
  public:
    using algorithm_def_tl_trampoline< triangulate_landmarks_base>::
              algorithm_def_tl_trampoline;

    void
    triangulate( kwiver::vital::camera_map_sptr cameras,
                 kwiver::vital::feature_track_set_sptr tracks,
                 kwiver::vital::landmark_map_sptr& landmarks )  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::triangulate_landmarks,
        triangulate,
        cameras,
        tracks,
        landmarks
      );
    }

    void
    triangulate( kwiver::vital::camera_map_sptr cameras,
                 kwiver::vital::track_map_t tracks,
                 kwiver::vital::landmark_map_sptr& landmarks )  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::triangulate_landmarks,
        triangulate,
        cameras,
        tracks,
        landmarks
      );
    }

};

}
}
}

#endif
