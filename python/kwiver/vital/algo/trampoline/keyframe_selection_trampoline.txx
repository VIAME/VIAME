// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file keyframe_selection_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<keyframe_selection> and keyframe_selection
 */

#ifndef KEYFRAME_SELECTION_TRAMPOLINE_TXX
#define KEYFRAME_SELECTION_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/keyframe_selection.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_kf_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::keyframe_selection > >
class algorithm_def_kf_trampoline :
      public algorithm_trampoline<algorithm_def_kf_base>
{
  public:
    using algorithm_trampoline<algorithm_def_kf_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::keyframe_selection>,
        type_name,
      );
    }
};

template< class keyframe_selection_base=
                kwiver::vital::algo::keyframe_selection >
class keyframe_selection_trampoline :
      public algorithm_def_kf_trampoline< keyframe_selection_base >
{
  public:
    using algorithm_def_kf_trampoline< keyframe_selection_base>::
              algorithm_def_kf_trampoline;

    kwiver::vital::track_set_sptr
    select( kwiver::vital::track_set_sptr tracks ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::track_set_sptr,
        kwiver::vital::algo::keyframe_selection,
        select,
        tracks
      );
    }
};

}
}
}

#endif
