/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "plane_ref_debug.h"

#include <boost/io/ios_state.hpp>

#include <iomanip>
#include <ostream>

/**
 * \file plane_ref_debug.cxx
 *
 * \brief Implementation of reference plane debugging functions.
 */

namespace vistk
{

void
debug_plane_ref_write(std::ostream& ostr, plane_ref_t const& ref)
{
  // Number of nibbles in the reference plus two for the prefix.
  static int const ref_width = sizeof(plane_ref::reference_t) / 8 * 2 + 2;

  boost::io::ios_flags_saver ifs(ostr);

  (void)ifs;

  ostr << std::hex;
  ostr << std::showbase;

  ostr << "Reference: ";

  if (ref)
  {
    ostr << std::internal;
    ostr << std::setw(ref_width);
    ostr << std::setfill('0');

    ostr << ref->reference();
  }
  else
  {
    ostr << "(null)";
  }
}

}
