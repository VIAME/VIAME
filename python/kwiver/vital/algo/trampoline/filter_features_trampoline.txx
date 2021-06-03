// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file filter_features_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::filter_features \endlink
 */

#ifndef FILTER_FEATURES_TRAMPOLINE_TXX
#define FILTER_FEATURES_TRAMPOLINE_TXX

#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/filter_features.h>

#include <python/kwiver/vital/util/pybind11.h>
#include <pybind11/stl.h>

namespace kwiver {
namespace vital  {
namespace python {
template< class algorithm_def_ff_base=
           kwiver::vital::algorithm_def<
               kwiver::vital::algo::filter_features > >
class algorithm_def_ff_trampoline :
      public algorithm_trampoline< algorithm_def_ff_base >
{
  public:
    using algorithm_trampoline< algorithm_def_ff_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def< kwiver::vital::algo::filter_features >,
        type_name,
      );
    }
};

template< class filter_features_base =
                  kwiver::vital::algo::filter_features >
class filter_features_trampoline :
      public algorithm_def_ff_trampoline< filter_features_base >
{
  public:
    using algorithm_def_ff_trampoline< filter_features_base >::
              algorithm_def_ff_trampoline;

    kwiver::vital::feature_set_sptr
    filter( kwiver::vital::feature_set_sptr input ) const override
    {
      VITAL_PYBIND11_OVERLOAD(
        kwiver::vital::feature_set_sptr,
        kwiver::vital::algo::filter_features,
        filter,
        input
      );
    }

    /*
     * TODO: Bind feature set
    std::pair< kwiver::vital::feature_set_sptr,
               kwiver::vital::descriptor_set_sptr >
    filter( kwiver::vital::feature_set_sptr feat,
            kwiver::vital::descriptor_set_sptr descr ) const override
    {
      VITAL_PYBIND11_OVERLOAD(
        kwiver::vital::feature_set_sptr,
        kwiver::vital::algo::filter_features,
        filter,
        feat,
        descr
      );
    }
    */

    kwiver::vital::feature_set_sptr
    filter( kwiver::vital::feature_set_sptr feat,
            std::vector<unsigned int>& indices ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::feature_set_sptr,
        kwiver::vital::algo::filter_features,
        filter,
        feat,
        indices
      );
    }
};

}
}
}

#endif
