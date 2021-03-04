// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file close_loops_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::close_loops \endlink
 */

#ifndef CLOSE_LOOPS_TRAMPOLINE_TXX
#define CLOSE_LOOPS_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/close_loops.h>

#include <ostream>

namespace kwiver {
namespace vital  {
namespace python {

template< class algorithm_def_cl_base=
           kwiver::vital::algorithm_def< kwiver::vital::algo::close_loops > >
class algorithm_def_cl_trampoline :
      public algorithm_trampoline< algorithm_def_cl_base>
{
  public:
    using algorithm_trampoline< algorithm_def_cl_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::close_loops>,
        type_name,
      );
    }
};

template< class close_loops_base=kwiver::vital::algo::close_loops >
class close_loops_trampoline :
      public algorithm_def_cl_trampoline< close_loops_base >
{
  public:
    using algorithm_def_cl_trampoline< close_loops_base >::
              algorithm_def_cl_trampoline;

    kwiver::vital::feature_track_set_sptr
    stitch( kwiver::vital::frame_id_t frame_number,
           kwiver::vital::feature_track_set_sptr input,
           kwiver::vital::image_container_sptr image,
           kwiver::vital::image_container_sptr mask ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::feature_track_set_sptr,
        kwiver::vital::algo::close_loops,
        stitch,
        frame_number,
        input,
        image,
        mask
      );
    }
};
}
}
}
#endif
