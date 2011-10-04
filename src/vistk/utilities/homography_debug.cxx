/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "homography_debug.h"

#include "homography.h"

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
debug_homography_base_write(std::ostream& ostr, homography_base const& homog)
{
  static int const precision = 16;

  boost::io::ios_flags_saver ifs(ostr);

  (void)ifs;

  ostr << std::boolalpha;
  ostr << std::fixed;
  ostr << std::setprecision(precision);

  homography_base::transform_t const& t = homog.transform();

  ostr << "Valid:   " << homog.is_valid() << "\n"
          "New ref: " << homog.is_new_reference() << "\n"
          "  [ " << t.get(0, 0) << ", " << t.get(0, 1) << ", " << t.get(0, 2) << " ]\n"
          "  [ " << t.get(1, 0) << ", " << t.get(1, 1) << ", " << t.get(1, 2) << " ]\n"
          "  [ " << t.get(2, 0) << ", " << t.get(2, 1) << ", " << t.get(2, 2) << " ]";
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
