// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file integrate_depth_maps_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<integrate_depth_maps> and integrate_depth_maps
 */

#ifndef INTEGRATE_DEPTH_MAPS_TRAMPOLINE_TXX
#define INTEGRATE_DEPTH_MAPS_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/integrate_depth_maps.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_idm_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::integrate_depth_maps > >
class algorithm_def_idm_trampoline :
      public algorithm_trampoline<algorithm_def_idm_base>
{
  public:
    using algorithm_trampoline<algorithm_def_idm_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::integrate_depth_maps>,
        type_name,
      );
    }
};

template< class integrate_depth_maps_base=
                kwiver::vital::algo::integrate_depth_maps >
class integrate_depth_maps_trampoline :
      public algorithm_def_idm_trampoline< integrate_depth_maps_base >
{
  public:
    using algorithm_def_idm_trampoline< integrate_depth_maps_base>::
              algorithm_def_idm_trampoline;

    void
    integrate( kwiver::vital::vector_3d const& minpt_bound,
               kwiver::vital::vector_3d const& maxpt_bound,
               std::vector<kwiver::vital::image_container_sptr> const& depth_maps,
               std::vector<kwiver::vital::camera_perspective_sptr> const& cameras,
               kwiver::vital::image_container_sptr& volume,
               kwiver::vital::vector_3d& spacing ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::integrate_depth_maps,
        integrate,
        minpt_bound,
        maxpt_bound,
        depth_maps,
        cameras,
        volume,
        spacing
      );
    }

    void
    integrate( kwiver::vital::vector_3d const& minpt_bound,
               kwiver::vital::vector_3d const& maxpt_bound,
               std::vector<kwiver::vital::image_container_sptr> const& depth_maps,
               std::vector<kwiver::vital::image_container_sptr> const& weight_maps,
               std::vector<kwiver::vital::camera_perspective_sptr> const& cameras,
               kwiver::vital::image_container_sptr& volume,
               kwiver::vital::vector_3d& spacing ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::integrate_depth_maps,
        integrate,
        minpt_bound,
        maxpt_bound,
        depth_maps,
        weight_maps,
        cameras,
        volume,
        spacing
      );
    }
};

}
}
}

#endif
