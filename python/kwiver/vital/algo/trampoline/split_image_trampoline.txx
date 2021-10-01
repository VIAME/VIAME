// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file split_image_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<split_image> and split_image
 */

#ifndef SPLIT_IMAGE_TRAMPOLINE_TXX
#define SPLIT_IMAGE_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/split_image.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_si_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::split_image > >
class algorithm_def_si_trampoline :
      public algorithm_trampoline<algorithm_def_si_base>
{
  public:
    using algorithm_trampoline<algorithm_def_si_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::split_image>,
        type_name,
      );
    }
};

template< class split_image_base=
                kwiver::vital::algo::split_image >
class split_image_trampoline :
      public algorithm_def_si_trampoline< split_image_base >
{
  public:
    using algorithm_def_si_trampoline< split_image_base>::
              algorithm_def_si_trampoline;

    std::vector< kwiver::vital::image_container_sptr >
    split( kwiver::vital::image_container_sptr image_data ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        std::vector< kwiver::vital::image_container_sptr >,
        kwiver::vital::algo::split_image,
        split,
        image_data
      );
    }
};

}
}
}

#endif
