// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file detect_motion_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::detect_motion \endlink
 */

#ifndef DETECT_MOTION_TRAMPOLINE_TXX
#define DETECT_MOTION_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/detect_motion.h>

namespace kwiver {
namespace vital  {
namespace python {
template< class algorithm_def_dm_base=
           kwiver::vital::algorithm_def< kwiver::vital::algo::detect_motion > >
class algorithm_def_dm_trampoline :
      public algorithm_trampoline< algorithm_def_dm_base>
{
  public:
    using algorithm_trampoline< algorithm_def_dm_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::detect_motion>,
        type_name,
      );
    }
};

template< class detect_motion_base=kwiver::vital::algo::detect_motion >
class detect_motion_trampoline :
      public algorithm_def_dm_trampoline< detect_motion_base >
{
  public:
    using algorithm_def_dm_trampoline< detect_motion_base >::
              algorithm_def_dm_trampoline;

    kwiver::vital::image_container_sptr
    process_image ( kwiver::vital::timestamp const& ts,
                    kwiver::vital::image_container_sptr const image,
                    bool reset_model ) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::image_container_sptr,
        kwiver::vital::algo::detect_motion,
        process_image,
        ts,
        image,
        reset_model
      );
    }
};
}
}
}
#endif
