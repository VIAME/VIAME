// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file detect_features_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::detect_features \endlink
 */

#ifndef DETECT_FEATURES_TRAMPOLINE_TXX
#define DETECT_FEATURES_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/detect_features.h>

namespace kwiver {
namespace vital  {
namespace python {

template< class algorithm_def_df_base=
           kwiver::vital::algorithm_def< kwiver::vital::algo::detect_features > >
class algorithm_def_df_trampoline :
      public algorithm_trampoline< algorithm_def_df_base>
{
  public:
    using algorithm_trampoline< algorithm_def_df_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::detect_features>,
        type_name,
      );
    }
};

template< class detect_features_base=kwiver::vital::algo::detect_features >
class detect_features_trampoline :
      public algorithm_def_df_trampoline< detect_features_base >
{
  public:
    using algorithm_def_df_trampoline< detect_features_base >::
              algorithm_def_df_trampoline;

    kwiver::vital::feature_set_sptr
    detect( kwiver::vital::image_container_sptr image_data,
            kwiver::vital::image_container_sptr mask ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::feature_set_sptr,
        kwiver::vital::algo::detect_features,
        detect,
        image_data,
        mask
      );
    }
};
}
}
}
#endif
