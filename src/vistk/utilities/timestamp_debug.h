/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_UTILITIES_TIMESTAMP_DEBUG_H
#define VISTK_UTILITIES_TIMESTAMP_DEBUG_H

#include "utilities-config.h"

#include <iosfwd>

/**
 * \file timestamp_debug.h
 *
 * \brief Debugging functions for timestamps.
 */

namespace vistk
{

class timestamp;

}

namespace vistk
{

/**
 * \brief Output information about a timestamp.
 *
 * \param ostr The stream to output to.
 * \param ts The timestamp to output.
 */
void VISTK_UTILITIES_EXPORT debug_timestamp_write(std::ostream& ostr, timestamp const& ts);

}

#endif // VISTK_UTILITIES_TIMESTAMP_DEBUG_H
