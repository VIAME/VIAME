// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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

namespace kwiver {
namespace vital  {
namespace python {

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

    kwiver::vital::image_container_sptr
    compute( std::vector<kwiver::vital::image_container_sptr> const& frames,
              std::vector<kwiver::vital::camera_perspective_sptr> const& cameras,
              double depth_min, double depth_max,
              unsigned int reference_frame,
              kwiver::vital::bounding_box<int> const& roi,
              kwiver::vital::image_container_sptr& depth_uncertainty,
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
        depth_uncertainty,
        mask
      );
    }
};
}
}
}
#endif
