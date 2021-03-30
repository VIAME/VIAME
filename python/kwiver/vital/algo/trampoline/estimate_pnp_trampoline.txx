// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file estimate_homography_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::estimate_homography \endlink
 */

#ifndef ESTIMATE_PNP_TRAMPOLINE_TXX
#define ESTIMATE_PNP_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/estimate_pnp.h>
#include <vital/types/vector.h>

namespace kwiver {
namespace vital  {
namespace python {
template< class algorithm_def_epnp_base=
           kwiver::vital::algorithm_def<
               kwiver::vital::algo::estimate_pnp > >
class algorithm_def_epnp_trampoline :
      public algorithm_trampoline< algorithm_def_epnp_base >
{
  public:
    using algorithm_trampoline< algorithm_def_epnp_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<
         kwiver::vital::algo::estimate_pnp >,
        type_name,
      );
    }
};

template< class estimate_pnp_base =
                  kwiver::vital::algo::estimate_pnp >
class estimate_pnp_trampoline :
      public algorithm_def_epnp_trampoline< estimate_pnp_base >
{
  public:
    using algorithm_def_epnp_trampoline< estimate_pnp_base >::
              algorithm_def_epnp_trampoline;

    kwiver::vital::camera_perspective_sptr
      estimate( const std::vector<kwiver::vital::vector_2d>& pts2d,
                const std::vector<kwiver::vital::vector_3d>& pts3d,
                const kwiver::vital::camera_intrinsics_sptr cal,
                std::vector<bool>& inliers )  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::camera_perspective_sptr,
        kwiver::vital::algo::estimate_pnp,
        estimate,
        pts2d,
        pts3d,
        cal,
        inliers
      );
    }
};
}
}
}

#endif
