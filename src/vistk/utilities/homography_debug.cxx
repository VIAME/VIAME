/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "homography_debug.h"

#include <boost/io/ios_state.hpp>

#include <iomanip>
#include <ostream>

/**
 * \file homography_debug.cxx
 *
 * \brief Implementation of the homography debugging functions.
 */

namespace vistk
{

void
debug_transform_write(std::ostream& ostr, homography_base::transform_t const& transform)
{
  static int const precision = 16;

  homography_base::transform_t const& t = transform;

  boost::io::ios_flags_saver const ifs(ostr);

  (void)ifs;

  ostr << std::fixed;
  ostr << std::setprecision(precision);

  ostr << "  [ " << t.get(0, 0) << ", " << t.get(0, 1) << ", " << t.get(0, 2) << " ]\n"
          "  [ " << t.get(1, 0) << ", " << t.get(1, 1) << ", " << t.get(1, 2) << " ]\n"
          "  [ " << t.get(2, 0) << ", " << t.get(2, 1) << ", " << t.get(2, 2) << " ]";
}

void
debug_homography_base_write(std::ostream& ostr, homography_base const& homog)
{
  homography_base::transform_t const& transform = homog.transform();

  boost::io::ios_flags_saver const ifs(ostr);

  (void)ifs;

  ostr << std::boolalpha;

  ostr << "Valid:   " << homog.is_valid() << "\n"
          "New ref: " << homog.is_new_reference() << "\n";

  debug_transform_write(ostr, transform);
}

template <typename Source, typename Dest>
void
debug_homography_write(std::ostream& ostr, homography<Source, Dest> const& homog)
{
  ostr << "Source: " << homog.source() << "\n"
          "Dest:   " << homog.dest() << "\n";

  debug_homography_base_write(ostr, homog);
}

}
