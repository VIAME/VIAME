/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "utm.h"

/**
 * \file utm.cxx
 *
 * \brief Implementation of utm data structures.
 */

namespace vistk
{

utm_zone_t::zone_t const utm_zone_t::zone_invalid = zone_t(-1);

utm_zone_t
::utm_zone_t()
  : m_zone(zone_invalid)
  , m_hemisphere(hemi_default)
{
}

utm_zone_t
::utm_zone_t(zone_t z, hemisphere_t h)
  : m_zone(z)
  , m_hemisphere(h)
{
}

utm_zone_t
::~utm_zone_t()
{
}

utm_zone_t::zone_t
utm_zone_t
::zone() const
{
  return m_zone;
}

utm_zone_t::hemisphere_t
utm_zone_t
::hemisphere() const
{
  return m_hemisphere;
}

void
utm_zone_t
::set_zone(zone_t z)
{
  m_zone = z;
}

void
utm_zone_t
::set_hemisphere(hemisphere_t h)
{
  m_hemisphere = h;
}

bool
utm_zone_t
::operator == (utm_zone_t const& utm) const
{
  return ((m_zone == utm.m_zone) &&
          (m_hemisphere == utm.m_hemisphere));
}

}
