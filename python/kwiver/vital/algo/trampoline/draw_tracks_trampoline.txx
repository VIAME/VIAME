// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file draw_tracks_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::draw_tracks \endlink
 */

#ifndef DETECT_TRACKS_TRAMPOLINE_TXX
#define DETECT_TRACKS_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/draw_tracks.h>

namespace kwiver {
namespace vital  {
namespace python {

template< class algorithm_def_dt_base=
           kwiver::vital::algorithm_def< kwiver::vital::algo::draw_tracks > >
class algorithm_def_dt_trampoline :
      public algorithm_trampoline< algorithm_def_dt_base>
{
  public:
    using algorithm_trampoline< algorithm_def_dt_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::draw_tracks>,
        type_name,
      );
    }
};

template< class draw_tracks_base=kwiver::vital::algo::draw_tracks >
class draw_tracks_trampoline :
      public algorithm_def_dt_trampoline< draw_tracks_base >
{
  public:
    using algorithm_def_dt_trampoline< draw_tracks_base >::
              algorithm_def_dt_trampoline;

    kwiver::vital::image_container_sptr
    draw( kwiver::vital::track_set_sptr display_set,
          kwiver::vital::image_container_sptr_list image_data,
          kwiver::vital::track_set_sptr comparision_set ) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::image_container_sptr,
        kwiver::vital::algo::draw_tracks,
        draw,
        display_set,
        image_data,
        comparision_set
      );
    }
};
}
}
}

#endif
