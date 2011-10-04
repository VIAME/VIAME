/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_UTILITIES_UTM_DEBUG_H
#define VISTK_UTILITIES_UTM_DEBUG_H

#include "utilities-config.h"

#include <iosfwd>

/**
 * \file utm_debug.h
 *
 * \brief Debugging functions for utm structures.
 */

namespace vistk
{

class utm_zone_t;

}

namespace vistk
{

/**
 * \brief Output information about a utm zone.
 *
 * \param ostr The stream to output to.
 * \param ref The zone to output.
 */
void VISTK_UTILITIES_EXPORT debug_utm_zone_write(std::ostream& ostr, utm_zone_t const& utm);

}

#endif // VISTK_UTILITIES_UTM_DEBUG_H
