/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_UTILITIES_PLANE_REF_DEBUG_H
#define VISTK_UTILITIES_PLANE_REF_DEBUG_H

#include "utilities-config.h"

#include "plane_ref.h"

#include <ostream>

/**
 * \file plane_ref_debug.h
 *
 * \brief Debugging functions for reference planes.
 */

namespace vistk
{

/**
 * \brief Output information about a reference plane.
 *
 * \param ostr The stream to output to.
 * \param ref The reference plane to output.
 */
void VISTK_UTILITIES_EXPORT debug_plane_ref_write(std::ostream& ostr, plane_ref_t const& ref);

}

#endif // VISTK_UTILITIES_PLANE_REF_DEBUG_H
