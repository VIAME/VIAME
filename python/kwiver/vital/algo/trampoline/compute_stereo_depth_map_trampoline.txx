// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file compute_stereo_depth_map_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::compute_stereo_depth_map \endlink
 */

#ifndef COMPUTE_STEREO_DEPTH_MAP_TXX
#define COMPUTE_STEREO_DEPTH_MAP_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/compute_stereo_depth_map.h>

namespace kwiver {
namespace vital  {
namespace python {

template< class algorithm_def_csdm_base=
            kwiver::vital::algorithm_def<
               kwiver::vital::algo::compute_stereo_depth_map > >
class algorithm_def_csdm_trampoline :
      public algorithm_trampoline<algorithm_def_csdm_base>
{
  public:
    using algorithm_trampoline< algorithm_def_csdm_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<
          kwiver::vital::algo::compute_stereo_depth_map>,
        type_name,
      );
    }
};

template< class compute_stereo_depth_map_base=
                  kwiver::vital::algo::compute_stereo_depth_map >
class compute_stereo_depth_map_trampoline :
      public algorithm_def_csdm_trampoline< compute_stereo_depth_map_base >
{
  public:
    using algorithm_def_csdm_trampoline< compute_stereo_depth_map_base >::
              algorithm_def_csdm_trampoline;

    kwiver::vital::image_container_sptr
    compute( kwiver::vital::image_container_sptr left_image,
             kwiver::vital::image_container_sptr right_image )
         const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::image_container_sptr,
        kwiver::vital::algo::compute_stereo_depth_map,
        compute,
        left_image,
        right_image
      );
    }
};
}
}
}
#endif
