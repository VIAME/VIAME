// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file image_filter_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<image_filter> and image_filter
 */

#ifndef IMAGE_FILTER_TRAMPOLINE_TXX
#define IMAGE_FILTER_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/image_filter.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/image_container.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_if_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::image_filter > >
class algorithm_def_if_trampoline :
      public algorithm_trampoline<algorithm_def_if_base>
{
  public:
    using algorithm_trampoline<algorithm_def_if_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::image_filter>,
        type_name,
      );
    }
};

template< class image_filter_base=kwiver::vital::algo::image_filter >
class image_filter_trampoline :
      public algorithm_def_if_trampoline< image_filter_base >
{
  public:
    using algorithm_def_if_trampoline< image_filter_base>::
              algorithm_def_if_trampoline;

    kwiver::vital::image_container_sptr
      filter( kwiver::vital::image_container_sptr data ) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::image_container_sptr,
        kwiver::vital::algo::image_filter,
        filter,
        data
      );
    }
};

}
}
}

#endif
