/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
