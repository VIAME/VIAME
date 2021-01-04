// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file convert_image_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::convert_image \endlink
 */

#ifndef COVERT_IMAGE_TRAMPOLINE_TXX
#define COVERT_IMAGE_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/convert_image.h>

namespace kwiver {
namespace vital  {
namespace python {

template< class algorithm_def_ci_base=
           kwiver::vital::algorithm_def< kwiver::vital::algo::convert_image > >
class algorithm_def_ci_trampoline :
      public algorithm_trampoline< algorithm_def_ci_base>
{
  public:
    using algorithm_trampoline< algorithm_def_ci_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::convert_image>,
        type_name,
      );
    }
};

template< class convert_image_base=kwiver::vital::algo::convert_image >
class convert_image_trampoline :
      public algorithm_def_ci_trampoline< convert_image_base >
{
  public:
    using algorithm_def_ci_trampoline< convert_image_base >::
              algorithm_def_ci_trampoline;

    kwiver::vital::image_container_sptr
    convert( kwiver::vital::image_container_sptr img ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::image_container_sptr,
        kwiver::vital::algo::convert_image,
        convert,
        img
      );
    }
};
}
}
}
#endif
