/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "utm_debug.h"

#include "utm.h"

#include <ostream>
#include <string>

/**
 * \file utm_debug.cxx
 *
 * \brief Implementation of utm data structure debugging classes.
 */

namespace vistk
{

static std::string const north_str = "north";
static std::string const south_str = "south";
static std::string const unknown_str = "unknown";

void
debug_utm_zone_write(std::ostream& ostr, utm_zone_t const& utm)
{
  std::string hemi_str;
  utm_zone_t::hemisphere_t const hemi = utm.hemisphere();

  switch (hemi)
  {
    case utm_zone_t::hemi_north:
      hemi_str = north_str;
      break;
    case utm_zone_t::hemi_south:
      hemi_str = south_str;
      break;
    default:
      hemi_str = unknown_str;
      break;
  }

  ostr << "Zone: " << utm.zone() << " Hemisphere: " << hemi_str;
}

}
