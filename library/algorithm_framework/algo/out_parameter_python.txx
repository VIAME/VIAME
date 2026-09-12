// This file is part of VIAME, and is distributed under an OSI-approved
// BSD 3-Clause License. See either the root top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

/// \file
/// \brief Read a python implementation's return value into C++ out parameters
///
/// Shared by the hand-written trampolines in `trampolines/`; see the README
/// there for why they exist at all.

#ifndef KWIVER_PYTHON_ALGO_OUT_PARAMETER_TXX
#define KWIVER_PYTHON_ALGO_OUT_PARAMETER_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/logger/logger.h>

#include <tuple>
#include <utility>

namespace kwiver::vital::python {

/// Read `result` as the return value followed by the out parameters.
///
/// The convention a python implementation of one of these methods follows:
/// return a tuple of what the C++ signature returns, then each out parameter
/// in the order the signature declares them. So an `extract_descriptors`
/// written in python returns `(descriptors, features)`, and an
/// `estimate_homography` returns `(homography, inliers)`.
///
/// Returning the bare value is still accepted, because that is what an
/// implementation written before this convention existed does, and because a
/// tuple requirement enforced by a cast error would say nothing useful. The
/// out parameters keep the values the caller passed in, and a warning says
/// which method left them alone -- once, per method, since these are called
/// per frame.
///
/// \param result   what the python method returned
/// \param what     the method's name, for the warning
/// \param outs     the C++ out parameters, in signature order
template < typename Return, typename... Outs >
Return
unpack_out_parameters(
  pybind11::object result, char const* what, Outs&... outs )
{
  constexpr size_t expected = 1 + sizeof...( Outs );

  if( !pybind11::isinstance< pybind11::tuple >( result ) )
  {
    static bool warned = false;
    if( !warned )
    {
      warned = true;
      LOG_WARN(
        kwiver::vital::get_logger( "python.algo" ),
        what << " returned a single value where this build expects "
             << expected << " (the return value and then each output "
                "parameter). Its output parameters are left as they were "
                "passed in, which for an inlier vector means no point is an "
                "inlier and for a feature set means the features no longer "
                "line up with the descriptors." );
    }

    return result.template cast< Return >();
  }

  auto values = result.template cast< pybind11::tuple >();

  if( values.size() != expected )
  {
    throw pybind11::value_error(
      std::string( what ) + " returned " +
      std::to_string( values.size() ) + " values, expected " +
      std::to_string( expected ) +
      " (the return value and then each output parameter)" );
  }

  size_t index = 1;
  ( ( outs = values[ index++ ].template cast< Outs >() ), ... );

  return values[ 0 ].template cast< Return >();
}

} // namespace kwiver::vital::python

#undef KWIVER_PYBIND11_INCLUDE
#endif
