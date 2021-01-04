// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file draw_detected_object_set_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::draw_detected_object_set \endlink
 */

#ifndef DETECT_DETECTED_OBJECT_SET_TRAMPOLINE_TXX
#define DETECT_DETECTED_OBJECT_SET_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/draw_detected_object_set.h>

namespace kwiver {
namespace vital  {
namespace python {
template< class algorithm_def_ddos_base=
           kwiver::vital::algorithm_def< kwiver::vital::algo::draw_detected_object_set > >
class algorithm_def_ddos_trampoline :
      public algorithm_trampoline< algorithm_def_ddos_base>
{
  public:
    using algorithm_trampoline< algorithm_def_ddos_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::draw_detected_object_set>,
        type_name,
      );
    }
};

template< class draw_detected_object_set_base=kwiver::vital::algo::draw_detected_object_set >
class draw_detected_object_set_trampoline :
      public algorithm_def_ddos_trampoline< draw_detected_object_set_base >
{
  public:
    using algorithm_def_ddos_trampoline< draw_detected_object_set_base >::
              algorithm_def_ddos_trampoline;

    kwiver::vital::image_container_sptr
    draw( kwiver::vital::detected_object_set_sptr detected_set,
           kwiver::vital::image_container_sptr image_data ) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::image_container_sptr,
        kwiver::vital::algo::draw_detected_object_set,
        draw,
        detected_set,
        image_data
      );
    }
};
}
}
}
#endif
