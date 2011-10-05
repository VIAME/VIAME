/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_UTILITIES_HOMOGRAPHY_DEBUG_H
#define VISTK_UTILITIES_HOMOGRAPHY_DEBUG_H

#include "utilities-config.h"

#include "homography.h"

#include <iosfwd>

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
void VISTK_UTILITIES_EXPORT debug_transform_write(std::ostream& ostr, homography_base::transform_t const& transform);

/**
 * \brief Output information in the base homography class.
 *
 * \param ostr The stream to output to.
 * \param homog The homography to output.
 */
void VISTK_UTILITIES_EXPORT debug_homography_base_write(std::ostream& ostr, homography_base const& homog);

/**
 * \brief Output information in the homography class.
 *
 * \param ostr The stream to output to.
 * \param homog The homography to output.
 */
template <typename Source, typename Dest>
void VISTK_UTILITIES_EXPORT debug_homography_write(std::ostream& ostr, homography<Source, Dest> const& homog);

}

#endif // VISTK_UTILITIES_HOMOGRAPHY_DEBUG_H
