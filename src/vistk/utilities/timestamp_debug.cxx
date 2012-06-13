/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "timestamp_debug.h"

#include "timestamp.h"

#include <boost/io/ios_state.hpp>

#include <iomanip>
#include <ostream>
#include <string>

/**
 * \file timestamp_debug.cxx
 *
 * \brief Implementation of timestamp debugging functions.
 */

namespace vistk
{

void
debug_timestamp_write(std::ostream& ostr, timestamp const& ts)
{
  static int const precision = 3;
  static std::string const str_invalid = "<inv>";

  boost::io::ios_flags_saver const ifs(ostr);

  (void)ifs;

  ostr << std::fixed;
  ostr << std::setprecision(precision);

  ostr << "(t:";

  if (ts.has_time())
  {
    ostr << ts.time();
  }
  else
  {
    ostr << str_invalid;
  }

  ostr << ", f:";

  if (ts.has_frame())
  {
    ostr << ts.frame();
  }
  else
  {
    ostr << str_invalid;
  }

  ostr << ")";
}

}
