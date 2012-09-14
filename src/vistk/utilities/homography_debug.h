/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_UTILITIES_HOMOGRAPHY_DEBUG_H
#define VISTK_UTILITIES_HOMOGRAPHY_DEBUG_H

#include "utilities-config.h"

#include "homography.h"

#include <ostream>

/**
 * \file homography_debug.h
 *
 * \brief Debugging functions for homographies.
 */

namespace vistk
{

/**
 * \brief Output information in the transformation for a homography.
 *
 * \param ostr The stream to output to.
 * \param transform The transform to output.
 */
VISTK_UTILITIES_EXPORT void debug_transform_write(std::ostream& ostr, homography_base::transform_t const& transform);

/**
 * \brief Output information in the base homography class.
 *
 * \param ostr The stream to output to.
 * \param homog The homography to output.
 */
VISTK_UTILITIES_EXPORT void debug_homography_base_write(std::ostream& ostr, homography_base const& homog);

/**
 * \brief Output information in the homography class.
 *
 * \param ostr The stream to output to.
 * \param homog The homography to output.
 */
template <typename Source, typename Dest>
void
debug_homography_write(std::ostream& ostr, homography<Source, Dest> const& homog)
{
  ostr << "Source: " << homog.source() << "\n"
          "Dest:   " << homog.dest() << "\n";

  debug_homography_base_write(ostr, homog);
}

}

#endif // VISTK_UTILITIES_HOMOGRAPHY_DEBUG_H
