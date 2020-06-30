// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file estimate_homography_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::estimate_homography \endlink
 */

#ifndef ESTIMATE_HOMOGRAPHY_TRAMPOLINE_TXX
#define ESTIMATE_HOMOGRAPHY_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/estimate_homography.h>
#include <vital/types/vector.h>

namespace kwiver {
namespace vital  {
namespace python {
template< class algorithm_def_eh_base=
           kwiver::vital::algorithm_def<
               kwiver::vital::algo::estimate_homography > >
class algorithm_def_eh_trampoline :
      public algorithm_trampoline< algorithm_def_eh_base>
{
  public:
    using algorithm_trampoline< algorithm_def_eh_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<
         kwiver::vital::algo::estimate_homography >,
        type_name,
      );
    }
};

template< class estimate_homography_base =
                  kwiver::vital::algo::estimate_homography >
class estimate_homography_trampoline :
      public algorithm_def_eh_trampoline< estimate_homography_base >
{
  public:
    using algorithm_def_eh_trampoline< estimate_homography_base >::
              algorithm_def_eh_trampoline;

    kwiver::vital::homography_sptr
      estimate(const kwiver::vital::feature_set_sptr feat1,
               const kwiver::vital::feature_set_sptr feat2,
               const kwiver::vital::match_set_sptr matches,
               std::vector<bool>& inliers,
               double inlier_scale)  const override
    {
      kwiver::vital::python::gil_scoped_acquire gil;
      pybind11::function overload = pybind11::get_overload( static_cast< const kwiver::vital::algo::estimate_homography* > ( this ), "estimate" );
      if( overload )
      {
        auto o = overload( feat1, feat2, matches, inlier_scale );
        if( o.is_none() )
        {
          return nullptr;
        }
        auto r = o.cast< std::pair< kwiver::vital::homography_sptr, std::vector<bool> > >();
        inliers = std::move( r.second );
        return r.first;
      }
      pybind11::pybind11_fail( "Tried to call pure virtual function \"kwiver::vital::algo::estimate_homography::estimate\"" );
    }

    kwiver::vital::homography_sptr
      estimate(const std::vector<kwiver::vital::vector_2d>& pts1,
               const std::vector<kwiver::vital::vector_2d>& pts2,
               std::vector<bool>& inliers,
               double inlier_scale)  const override
    {
      kwiver::vital::python::gil_scoped_acquire gil;
      pybind11::function overload = pybind11::get_overload( static_cast< const kwiver::vital::algo::estimate_homography* > ( this ), "estimate" );
      if( overload )
      {
        auto o = overload( pts1, pts2, inlier_scale );
        if( o.is_none() )
        {
          return nullptr;
        }
        auto r = o.cast< std::pair< kwiver::vital::homography_sptr, std::vector<bool> > >();
        inliers = std::move( r.second );
        return r.first;
      }
      pybind11::pybind11_fail( "Tried to call pure virtual function \"kwiver::vital::algo::estimate_homography::estimate\"" );
    }

};
}
}
}

#endif
