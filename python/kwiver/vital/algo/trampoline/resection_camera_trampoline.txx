// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file resection_camera_trampoline.txx
///
/// \brief trampoline for overriding virtual functions for
///        \link kwiver::vital::algo::resection_camera \endlink

#ifndef RESECTION_CAMERA_TRAMPOLINE_TXX
#define RESECTION_CAMERA_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/resection_camera.h>
#include <vital/types/vector.h>

namespace kwiver {
namespace vital  {
namespace python {
template< class algorithm_def_resection_base=
           kwiver::vital::algorithm_def<
               kwiver::vital::algo::resection_camera > >
class algorithm_def_resection_trampoline :
      public algorithm_trampoline< algorithm_def_resection_base >
{
  public:
    using algorithm_trampoline< algorithm_def_resection_base >
      ::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<
          kwiver::vital::algo::resection_camera >,
        type_name,
      );
    }
};

template< class resection_camera_base =
                  kwiver::vital::algo::resection_camera >
class resection_camera_trampoline :
      public algorithm_def_resection_trampoline< resection_camera_base >
{
  public:
    using algorithm_def_resection_trampoline< resection_camera_base >::
              algorithm_def_resection_trampoline;

    kwiver::vital::camera_perspective_sptr
      resection( const std::vector<kwiver::vital::vector_2d>& image_points,
                 const std::vector<kwiver::vital::vector_3d>& world_points,
                 kwiver::vital::camera_intrinsics_sptr cal,
                 std::vector<bool>* inliers
               ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::camera_perspective_sptr,
        kwiver::vital::algo::resection_camera,
        resection,
        image_points,
        world_points,
        cal,
        inliers
      );
    }

    kwiver::vital::camera_perspective_sptr
      resection( kwiver::vital::frame_id_t frame_id,
                 kwiver::vital::landmark_map_sptr landmarks,
                 kwiver::vital::feature_track_set_sptr tracks,
                 unsigned width, unsigned height,
                 std::unordered_set<landmark_id_t>* inliers
               ) const override
    {
      VITAL_PYBIND11_OVERLOAD(
        kwiver::vital::camera_perspective_sptr,
        kwiver::vital::algo::resection_camera,
        resection,
        frame_id,
        landmarks,
        tracks,
        width, height,
        inliers
      );
    }

    kwiver::vital::camera_perspective_sptr
      resection( kwiver::vital::frame_id_t frame_id,
                 kwiver::vital::landmark_map_sptr landmarks,
                 kwiver::vital::feature_track_set_sptr tracks,
                 kwiver::vital::camera_intrinsics_sptr cal,
                 std::unordered_set<landmark_id_t>* inliers
               ) const override
    {
      VITAL_PYBIND11_OVERLOAD(
        kwiver::vital::camera_perspective_sptr,
        kwiver::vital::algo::resection_camera,
        resection,
        frame_id,
        landmarks,
        tracks,
        cal,
        inliers
      );
    }
};
}
}
}

#endif
